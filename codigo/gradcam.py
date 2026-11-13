"""
gradcam.py — Grad-CAM para los modelos de segmentación del benchmark.

QUÉ HACE
    Genera un mapa de calor que indica qué regiones de la radiografía
    empujaron la predicción de lesión, y lo contrasta con la máscara GT y,
    cuando existe, con la caja TBX11K.

CÓMO
    1. Engancha la última salida del encoder (mapa de rasgos A, C×H'×W').
    2. Hace forward y calcula un score escalar: suma de logits dentro de la
       máscara GT (si existe) o de la máscara predicha (si el GT está vacío).
    3. Backward: gradiente del score respecto de A.
    4. Pesos α_k = media global del gradiente por canal; CAM = ReLU(Σ α_k A_k).
    5. Upsample bilineal a 512×512 y normalización min–max.

MÉTRICAS
    - Masa del CAM dentro del GT / masa total (0,5 = azar si el GT ocupa la
      mitad; el valor esperado por azar es exactamente el área del GT).
    - Lift = masa_en_GT − área_GT (positivo = el modelo mira la lesión).
    - Precisión del 20 % de píxeles con mayor CAM dentro del GT.
    - Si hay cajas: masa_en_caja y lift respecto del área de la caja.

USO
    python codigo/gradcam.py --modelo deeplabv3p --semilla 42 --limite 1
    python codigo/gradcam.py --modelo deeplabv3p --semilla 42 --indices 0,3,7
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_tb import DatasetTB, _normalizar, _ruta_local   # noqa: E402
from evaluar_test import cargar_modelo                         # noqa: E402

UMBRAL = 0.5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANOTACIONES = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K",
                           "annotations", "json",
                           "TBX11K_trainval_only_tb.json")


# --------------------------------------------------------------------------
# Cajas TBX11K (opcionales)
# --------------------------------------------------------------------------
def cargar_cajas(ruta=ANOTACIONES):
    """Devuelve {id_imagen: [(x1,y1,x2,y2), ...]} desde el JSON COCO."""
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in d.get("images", [])}
    cajas = {}
    for a in d.get("annotations", []):
        nombre = id2file.get(a["image_id"])
        if not nombre:
            continue
        clave = os.path.basename(nombre).rsplit(".", 1)[0]
        x, y, w, h = a["bbox"]
        cajas.setdefault(clave, []).append((x, y, x + w, y + h))
    return cajas


def mascara_cajas(cajas, forma):
    m = np.zeros(forma, dtype=bool)
    for x1, y1, x2, y2 in cajas:
        m[int(max(0, y1)):int(min(forma[0], y2)),
          int(max(0, x1)):int(min(forma[1], x2))] = True
    return m


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
class _Captura:
    """Hook que guarda el último tensor 4D de la capa y su gradiente."""

    def __init__(self, modulo):
        self.salida = None
        self.grad = None
        self._h = modulo.register_forward_hook(self._forward)

    def _forward(self, _mod, _inp, out):
        t = _ultimo_4d(out)
        if t is None:
            return
        self.salida = t
        if t.requires_grad:
            t.register_hook(self._backward)

    def _backward(self, grad):
        self.grad = grad

    def cerrar(self):
        self._h.remove()


def _ultimo_4d(obj):
    if isinstance(obj, torch.Tensor) and obj.ndim == 4:
        return obj
    if isinstance(obj, (list, tuple)):
        for t in reversed(obj):
            if isinstance(t, torch.Tensor) and t.ndim == 4:
                return t
    return None


def _capa_objetivo(net):
    for nombre in ("encoder", "swinViT", "backbone"):
        if hasattr(net, nombre):
            return getattr(net, nombre), nombre
    return net, "modelo"


def _logits(net, x):
    salida = net(x)
    if isinstance(salida, (tuple, list)):
        for s in salida:
            if isinstance(s, torch.Tensor) and s.ndim == 4:
                return s
        raise TypeError("salida del modelo sin tensor 4D")
    return salida


def grad_cam(net, img_tensor, mascara, device):
    """Devuelve (cam 512×512 en [0,1], logits 512×512, capa usada)."""
    capa, nombre_capa = _capa_objetivo(net)
    cap = _Captura(capa)
    net.zero_grad(set_to_none=True)
    x = img_tensor[None, None].to(device)

    logits = _logits(net, x)[0, 0]
    m = mascara
    if m.sum() == 0:
        m = (torch.sigmoid(logits.detach()) > UMBRAL).cpu().numpy()
    if m.sum() == 0:
        score = logits.mean()
    else:
        score = logits[torch.from_numpy(m).to(device)].sum()
    score.backward()

    acts = _ultimo_4d(cap.salida)
    grads = _ultimo_4d(cap.grad)
    cap.cerrar()
    if acts is None or grads is None:
        raise RuntimeError(
            f"la capa '{nombre_capa}' no produjo un tensor 4D con gradiente; "
            "usa un modelo con encoder convolucional o ajusta _capa_objetivo")

    pesos = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((pesos * acts).sum(dim=1))[0]
    cam = F.interpolate(cam[None, None], size=logits.shape[-2:],
                        mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, logits.detach().cpu().numpy(), nombre_capa


def metricas_cam(cam, gt, cajas=None):
    total = float(cam.sum())
    area_gt = float(gt.mean())
    masa_gt = float(cam[gt].sum() / total) if total > 0 and gt.any() else None
    out = {
        "masa_en_gt": None if masa_gt is None else round(masa_gt, 6),
        "area_gt": round(area_gt, 6),
        "lift_gt": None if masa_gt is None else round(masa_gt - area_gt, 6),
    }
    k = max(1, int(0.20 * cam.size))
    umbral_top = np.partition(cam.ravel(), -k)[-k]
    top = cam >= umbral_top
    out["precision_top20"] = round(float((top & gt).sum() / top.sum()), 6)

    if cajas:
        mc = mascara_cajas(cajas, gt.shape)
        out["masa_en_caja"] = round(float(cam[mc].sum() / total), 6) \
            if total > 0 and mc.any() else None
        out["area_caja"] = round(float(mc.mean()), 6)
        out["lift_caja"] = (None if out["masa_en_caja"] is None
                            else round(out["masa_en_caja"] - mc.mean(), 6))
    return out


# --------------------------------------------------------------------------
# Visualización
# --------------------------------------------------------------------------
def guardar_figura(ruta, img_raw, gt, cam, logits, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = prob > UMBRAL
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(img_raw, cmap="gray")
    ax[0].set_title("Radiografía")
    ax[1].imshow(img_raw, cmap="gray")
    if gt.any():
        ax[1].contour(gt, colors="lime", linewidths=1.2)
    ax[1].set_title("GT de plata")
    ax[2].imshow(cam, cmap="jet")
    ax[2].set_title("Grad-CAM")
    ax[3].imshow(img_raw, cmap="gray")
    ax[3].imshow(cam, cmap="jet", alpha=0.45)
    if pred.any():
        ax[3].contour(pred, colors="cyan", linewidths=1.0)
    ax[3].set_title("Solape + predicción")
    for a in ax:
        a.axis("off")
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Grad-CAM validado contra GT/caja")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--semilla", type=int, required=True)
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--split", default="test_pos", choices=["val", "test_pos"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--indices", default=None,
                    help="índices separados por coma; por defecto 0")
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    rd = os.path.join(args.resultados, args.modelo, f"seed_{args.semilla}")
    ckpt = os.path.join(rd, "mejor_modelo.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"sin checkpoint: {ckpt}")

    ds = DatasetTB(args.split, esperar_gt=True, verbose=False)
    if args.indices:
        indices = [int(i) for i in args.indices.split(",") if i.strip()]
    elif args.limite is not None:
        indices = list(range(min(args.limite, len(ds))))
    else:
        indices = [0]

    net = cargar_modelo(args.modelo, ckpt, torch.device(args.device))
    cajas = cargar_cajas()
    salida = Path(args.salida or (Path(args.resultados) / "gradcam"))
    salida.mkdir(parents=True, exist_ok=True)

    filas = []
    for i in indices:
        fila = ds.filas[i]
        img = ds._imagen_desde_png(_ruta_local(fila["ruta"])) \
            if args.split == "test_pos" \
            else ds._imagen_desde_npz(_ruta_local(fila["ruta"]))
        gt = ds._cargar_mask(fila if args.split == "test_pos"
                             else _ruta_local(fila["ruta"])).astype(bool)
        img_n = _normalizar(img)
        cam, logits, capa = grad_cam(net, torch.from_numpy(img_n),
                                     gt, torch.device(args.device))
        m = metricas_cam(cam, gt, cajas.get(fila["id"]))
        m.update({"id": fila["id"], "indice": i, "capa": capa,
                  "modelo": args.modelo, "semilla": args.semilla})
        filas.append(m)
        png = salida / f"{args.modelo}_s{args.semilla}_{fila['id']}.png"
        guardar_figura(png, img, gt, cam, logits,
                       f"{args.modelo} · semilla {args.semilla} · {fila['id']}")
        print(f"  {fila['id']}: masa_en_gt={m['masa_en_gt']} "
              f"lift={m['lift_gt']} top20={m['precision_top20']} -> {png.name}")

    ruta_csv = salida / f"{args.modelo}_s{args.semilla}.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"  guardado: {ruta_csv}")


if __name__ == "__main__":
    main()
