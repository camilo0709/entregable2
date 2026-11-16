"""
CALIBRACIÓN, ROBUSTEZ E INTERPRETABILIDAD (Entregable 3, Bloque D)

Para un checkpoint (modelo, semilla):
  1. Calibración: ECE a nivel de píxel (15 bins) y Brier ANTES y DESPUÉS de
     escalado por temperatura. T se ajusta SOLO en validación (minimiza BCE);
     el test solo se mide. Guarda la tabla de confiabilidad.
  2. Robustez: Dice medio en test_pos bajo ruido gaussiano, desenfoque,
     contraste y brillo (en el espacio z-score de entrada).
  3. Grad-CAM: mapas sobre el último bloque del encoder (o la última conv si
     no hay encoder), objetivo = suma de logits en la región predicha.

Salidas en <salida>/<modelo>_s<semilla>/: calibracion.json, confiabilidad.csv,
robustez.csv, gradcam.png.
Uso: python analisis_extra.py --resultados R --modelo lacca_unet --semilla 42 \\
         --splits-dir datos/splits --salida R/extra
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PERTURBACIONES = [("ruido_0.1", ("ruido", 0.1)), ("ruido_0.3", ("ruido", 0.3)),
                  ("ruido_0.5", ("ruido", 0.5)), ("desenfoque_1", ("blur", 1.0)),
                  ("desenfoque_2", ("blur", 2.0)), ("contraste_0.7", ("contraste", 0.7)),
                  ("contraste_0.5", ("contraste", 0.5)),
                  ("brillo_+0.5", ("brillo", 0.5)), ("brillo_-0.5", ("brillo", -0.5))]


# ---------------------------------------------------------------- calibración
def ece_pixel(prob, gt, bins=15):
    """ECE y tabla de confiabilidad sobre vectores 1D de probabilidad y {0,1}."""
    prob, gt = np.asarray(prob, np.float64), np.asarray(gt, np.float64)
    bordes = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, bordes[1:-1]), 0, bins - 1)
    ece, filas = 0.0, []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            filas.append((b, bordes[b], bordes[b + 1], 0, float("nan"), float("nan")))
            continue
        conf, acc = float(prob[m].mean()), float(gt[m].mean())
        ece += (n / len(prob)) * abs(conf - acc)
        filas.append((b, bordes[b], bordes[b + 1], n, conf, acc))
    return float(ece), filas


def brier(prob, gt):
    return float(np.mean((np.asarray(prob, np.float64) - np.asarray(gt, np.float64)) ** 2))


def ajustar_temperatura(logits, gt, iters=200):
    """T > 0 que minimiza la BCE de sigmoid(logits / T). 1D, LBFGS sobre log T."""
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(gt), dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=iters)

    def cierre():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(z / logT.exp(), y)
        l.backward()
        return l
    opt.step(cierre)
    return float(logT.exp().item())


# ------------------------------------------------------------------ inferencia
@torch.no_grad()
def inferir(net, ds, device, batch=4, n_max=None, pix_por_img=None, semilla=0):
    """-> lista de (id, logits[H,W] np, gt[H,W] np). Submuestrea píxeles si se pide."""
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    salida = []
    for xb, yb, info in dl:
        lg = net(xb.to(device)).float().cpu()
        for k in range(lg.shape[0]):
            salida.append((info["id"][k], lg[k, 0].numpy(), yb[k, 0].numpy()))
        if n_max and len(salida) >= n_max:
            break
    return salida


def _aplanar(items, pix_por_img=20000, semilla=0):
    rng = np.random.default_rng(semilla)
    L, G = [], []
    for _, lg, gt in items:
        l, g = lg.ravel(), gt.ravel()
        if pix_por_img and len(l) > pix_por_img:
            i = rng.choice(len(l), pix_por_img, replace=False)
            l, g = l[i], g[i]
        L.append(l)
        G.append((g > 0.5).astype(np.float64))
    return np.concatenate(L), np.concatenate(G)


def calibrar(net, ds_val, ds_test_list, device):
    lv, gv = _aplanar(inferir(net, ds_val, device))
    T = ajustar_temperatura(lv, gv)
    lt, gt = _aplanar(sum([inferir(net, d, device) for d in ds_test_list], []))
    sig = lambda z: 1 / (1 + np.exp(-z))
    e0, tabla0 = ece_pixel(sig(lt), gt)
    e1, tabla1 = ece_pixel(sig(lt / T), gt)
    res = {"T": T, "ece_antes": e0, "ece_despues": e1,
           "brier_antes": brier(sig(lt), gt), "brier_despues": brier(sig(lt / T), gt),
           "n_pix_val": int(len(lv)), "n_pix_test": int(len(lt))}
    return res, tabla0, tabla1


# -------------------------------------------------------------------- robustez
def perturbar(x, tipo, v, semilla=0):
    if tipo == "ruido":
        g = torch.Generator().manual_seed(semilla)
        return x + v * torch.randn(x.shape, generator=g)
    if tipo == "blur":
        r = int(3 * v)
        t = torch.arange(-r, r + 1, dtype=x.dtype)
        k = torch.exp(-0.5 * (t / v) ** 2)
        k = (k / k.sum())
        y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    if tipo == "contraste":
        return x * v
    if tipo == "brillo":
        return x + v
    raise ValueError(tipo)


@torch.no_grad()
def robustez(net, ds, device, dice_fn, batch=4):
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    res = {}
    for nombre, (tipo, v) in [("limpio", ("brillo", 0.0))] + PERTURBACIONES:
        d = []
        for xb, yb, _ in dl:
            xp = perturbar(xb, tipo, v)
            pr = torch.sigmoid(net(xp.to(device))).cpu()
            for p_, y_ in zip(pr, yb):
                d.append(dice_fn(p_[0] > 0.5, y_[0] > 0.5))
        res[nombre] = float(np.mean(d))
    return res


# -------------------------------------------------------------------- Grad-CAM
def _capa_objetivo(net):
    base = getattr(net, "base", net)
    if hasattr(base, "encoder"):
        return base.encoder, "encoder"
    ultima = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    return ultima, "ultima_conv"


def gradcam(net, x):
    """x: (1,1,H,W) -> (cam[H,W] en 0..1, prob[H,W]). Requiere grad."""
    capa, modo = _capa_objetivo(net)
    guardado = {}

    def hook(_m, entrada, salida):
        t = salida[-1] if (modo == "encoder" and isinstance(salida, (list, tuple))) \
            else (salida if modo == "encoder" else entrada[0])
        t.retain_grad()
        guardado["a"] = t
    h = capa.register_forward_hook(hook)
    try:
        net.zero_grad()
        lg = net(x)
        pred = (torch.sigmoid(lg) > 0.5).float()
        objetivo = (lg * pred).sum() if pred.sum() > 0 else lg.mean()
        objetivo.backward()
    finally:
        h.remove()
    a = guardado["a"]
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy(), torch.sigmoid(lg)[0, 0].detach().numpy()


def figura_gradcam(net, ds, indices, device, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    net.eval()
    fig, ejes = plt.subplots(len(indices), 4, figsize=(11, 2.8 * len(indices)))
    ejes = np.atleast_2d(ejes)
    for r, i in enumerate(indices):
        x, y, info = ds[i]
        cam, prob = gradcam(net, x.unsqueeze(0).to(device))
        img = x[0].numpy()
        for c, (t, im, cm) in enumerate([("imagen", img, "gray"), ("GT", y[0].numpy(), "gray"),
                                         ("predicción", prob > 0.5, "gray"),
                                         ("Grad-CAM", img, "gray")]):
            ejes[r, c].imshow(im, cmap=cm)
            if t == "Grad-CAM":
                ejes[r, c].imshow(cam, cmap="jet", alpha=0.45)
            ejes[r, c].set_title(f"{info['id']} · {t}" if c == 0 else t, fontsize=8)
            ejes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--n-gradcam", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    from dataset_tb import DatasetTB
    from modelos_tb import crear_modelo
    import metricas_tb as M
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    ck = torch.load(os.path.join(a.resultados, a.modelo, f"seed_{a.semilla}",
                                 "mejor_modelo.pt"), map_location="cpu", weights_only=False)
    net, _ = crear_modelo(a.modelo, in_channels=1, out_channels=1, verbose=False,
                          params_extra=(ck.get("config") or {}).get("params_modelo"))
    net.load_state_dict(ck.get("state_dict", ck))
    net = net.to(device).eval()
    ds = lambda s: DatasetTB(s, splits_dir=a.splits_dir, augmentar=False, verbose=False)
    sal = os.path.join(a.salida or os.path.join(a.resultados, "extra"),
                       f"{a.modelo}_s{a.semilla}")
    os.makedirs(sal, exist_ok=True)

    res, t0, t1 = calibrar(net, ds("val"), [ds("test_pos"), ds("test_neg")], device)
    json.dump(res, open(os.path.join(sal, "calibracion.json"), "w"), indent=2)
    with open(os.path.join(sal, "confiabilidad.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "lo", "hi", "n_antes", "conf_antes", "acc_antes",
                    "n_despues", "conf_despues", "acc_despues"])
        for x, y in zip(t0, t1):
            w.writerow(list(x) + list(y[3:]))
    print(f"ECE {res['ece_antes']:.4f} -> {res['ece_despues']:.4f} (T={res['T']:.3f})")

    rob = robustez(net, ds("test_pos"), device, M.dice_pytorch)
    with open(os.path.join(sal, "robustez.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perturbacion", "dice_test_pos", "caida_vs_limpio"])
        for k, v in rob.items():
            w.writerow([k, round(v, 5), round(rob["limpio"] - v, 5)])
    print("robustez:", {k: round(v, 3) for k, v in rob.items()})

    dp = ds("test_pos")
    idx = np.linspace(0, len(dp) - 1, a.n_gradcam).astype(int).tolist()
    figura_gradcam(net, dp, idx, device, os.path.join(sal, "gradcam.png"))
    print("Grad-CAM:", os.path.join(sal, "gradcam.png"))


if __name__ == "__main__":
    main()
