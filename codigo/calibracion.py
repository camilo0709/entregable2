"""
calibracion.py — Calibración de probabilidades de los modelos de segmentación.

QUÉ MIDE
    Un modelo puede segmentar bien y, aun así, estar mal calibrado: decir 0,9
    cuando acierta el 70 % de las veces. Este módulo cuantifica esa brecha con
    Brier y ECE (Expected Calibration Error), globales y restringidos al
    primer plano (píxeles de lesión) y al fondo.

QUÉ AJUSTA
    Temperature scaling: un único parámetro T que divide los logits
    (z / T) y se ajusta SOLO sobre validación (nunca sobre test). No cambia
    el orden de las probabilidades ni la arquitectura; solo su escala.

PROTOCOLO
    - Se ajusta T sobre `val` (29 imágenes), jamás sobre `test_pos`.
    - El umbral de decisión del benchmark sigue fijo en 0,5: la calibración
      es un diagnóstico y una mejora posterior, no un cambio del preregistro.
    - El ECE se reporta con 15 bins (más fino que los 10 habituales) y con
      el número de píxeles por bin, para que ningún bin vacío infle la cifra.

USO
    python codigo/calibracion.py --modelo deeplabv3p --semilla 42
    python codigo/calibracion.py --modelo deeplabv3p --semilla 42 --split val --guardar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_tb import DatasetTB                       # noqa: E402
from evaluar_test import cargar_modelo                 # noqa: E402

BINS = 15


# --------------------------------------------------------------------------
# Métricas de calibración
# --------------------------------------------------------------------------
def brier(prob: np.ndarray, y: np.ndarray) -> float:
    """Error cuadrático medio entre probabilidad y etiqueta binaria."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean((prob - y) ** 2))


def ece(prob: np.ndarray, y: np.ndarray, bins: int = BINS):
    """Expected Calibration Error y tabla por bin.

    Devuelve (ece, tabla) donde tabla es una lista de dicts con
    bin, confianza media, acierto medio, n_pixeles y peso.
    """
    prob = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    bordes = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, bordes[1:-1]), 0, bins - 1)
    total = prob.size
    ece_val = 0.0
    tabla = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            tabla.append({"bin": b, "confianza": None, "acierto": None,
                          "n": 0, "peso": 0.0})
            continue
        conf = float(prob[m].mean())
        acc = float(y[m].mean())
        peso = n / total
        ece_val += peso * abs(acc - conf)
        tabla.append({"bin": b, "confianza": round(conf, 6),
                      "acierto": round(acc, 6), "n": n,
                      "peso": round(peso, 6)})
    return float(ece_val), tabla


def metricas_calibracion(prob: np.ndarray, y: np.ndarray, bins: int = BINS):
    """Brier/ECE global, en primer plano y en fondo."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    fg = y > 0.5
    out = {
        "brier_global": brier(prob, y),
        "ece_global": ece(prob, y, bins)[0],
        "n_pixeles": int(prob.size),
        "n_fg": int(fg.sum()),
        "frac_fg": float(fg.mean()),
    }
    if fg.any():
        out["brier_fg"] = brier(prob[fg], y[fg])
        out["ece_fg"] = ece(prob[fg], y[fg], bins)[0]
    if (~fg).any():
        out["brier_fondo"] = brier(prob[~fg], y[~fg])
        out["ece_fondo"] = ece(prob[~fg], y[~fg], bins)[0]
    return out


# --------------------------------------------------------------------------
# Temperature scaling
# --------------------------------------------------------------------------
def _nll_con_temperatura(logits: np.ndarray, y: np.ndarray, t: float) -> float:
    """BCE con logits divididos por T, calculada de forma estable."""
    z = np.asarray(logits, dtype=np.float64).ravel() / t
    y = np.asarray(y, dtype=np.float64).ravel()
    # log(1 + exp(z)) - y*z, estable para |z| grande
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def ajustar_temperatura(logits: np.ndarray, y: np.ndarray):
    """Ajusta T > 0 minimizando la NLL de validación.

    Se optimiza log(T) en [-2, 2] (T entre 0,135 y 7,39), un rango amplio
    pero sin riesgo de overflow. Devuelve (T, nll_antes, nll_despues).
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nll_antes = _nll_con_temperatura(logits, y, 1.0)

    def objetivo(u):
        return _nll_con_temperatura(logits, y, float(np.exp(u)))

    res = minimize_scalar(objetivo, bounds=(-2.0, 2.0), method="bounded",
                          options={"xatol": 1e-5})
    t = float(np.exp(res.x))
    return t, nll_antes, float(res.fun)


def aplicar_temperatura(logits: np.ndarray, t: float) -> np.ndarray:
    """Devuelve probabilidades sigmoid(logits / T)."""
    z = np.asarray(logits, dtype=np.float64) / float(t)
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------
def _a_logits(salida):
    """Normaliza la salida del modelo a un tensor (B,1,H,W)."""
    if isinstance(salida, (tuple, list)):
        for x in salida:
            if isinstance(x, torch.Tensor) and x.ndim == 4:
                return x
        raise TypeError("la salida del modelo no contiene un tensor 4D")
    return salida


@torch.no_grad()
def recoger_logits(net, ds, device, batch_size: int = 8, limite=None):
    """Devuelve (logits, mascaras) en float32/uint8, sin pasar por CPU de más."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, masks = [], []
    vistos = 0
    for imgs, msk, _ in dl:
        z = _a_logits(net(imgs.to(device))).detach().cpu().numpy()[:, 0]
        logits.append(z.astype(np.float32))
        masks.append((msk.numpy()[:, 0] > 0.5).astype(np.uint8))
        vistos += z.shape[0]
        if limite is not None and vistos >= limite:
            break
    logits = np.concatenate(logits, axis=0)
    masks = np.concatenate(masks, axis=0)
    if limite is not None:
        logits, masks = logits[:limite], masks[:limite]
    return logits, masks


def evaluar_calibracion(clave: str, semilla: int, res_dir: str,
                        split: str = "val", device: str = "cpu",
                        bins: int = BINS, limite=None):
    """Carga un checkpoint y evalúa calibración antes/después de T.

    T se ajusta SIEMPRE sobre `val`. Si `split` es `test_pos`, se ajusta en
    validación y se evalúa en test: así la temperatura no ve el test.
    """
    rd = os.path.join(res_dir, clave, f"seed_{semilla}")
    ckpt = os.path.join(rd, "mejor_modelo.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"sin checkpoint: {ckpt}")

    net = cargar_modelo(clave, ckpt, torch.device(device))
    ds_val = DatasetTB("val", esperar_gt=True, verbose=False)
    logits_val, y_val = recoger_logits(net, ds_val, torch.device(device),
                                       limite=limite)

    t, nll_antes, nll_despues = ajustar_temperatura(logits_val, y_val)

    if split == "val":
        logits, y = logits_val, y_val
    else:
        ds = DatasetTB(split, esperar_gt=True, verbose=False)
        logits, y = recoger_logits(net, ds, torch.device(device),
                                   limite=limite)

    prob_antes = aplicar_temperatura(logits, 1.0)
    prob_despues = aplicar_temperatura(logits, t)

    return {
        "modelo": clave,
        "semilla": semilla,
        "split": split,
        "ajuste_temperatura": "val",
        "n_imagenes": int(logits.shape[0]),
        "temperatura": round(t, 6),
        "nll_antes": round(nll_antes, 8),
        "nll_despues": round(nll_despues, 8),
        "antes": metricas_calibracion(prob_antes, y, bins),
        "despues": metricas_calibracion(prob_despues, y, bins),
        "tabla_bins": ece(prob_despues, y, bins)[1],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Calibración (Brier/ECE + temperatura)")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--semilla", type=int, required=True)
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--split", default="val", choices=["val", "test_pos"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--limite", type=int, default=None,
                    help="usar solo N imágenes (para pruebas rápidas)")
    ap.add_argument("--guardar", action="store_true")
    args = ap.parse_args()

    r = evaluar_calibracion(args.modelo, args.semilla, args.resultados,
                            args.split, args.device, args.bins, args.limite)
    print(f"\n{args.modelo} · semilla {args.semilla} · {args.split} "
          f"({r['n_imagenes']} imágenes)")
    print(f"  T óptima: {r['temperatura']:.4f}  "
          f"(NLL {r['nll_antes']:.6f} -> {r['nll_despues']:.6f})")
    print(f"  {'métrica':<14}{'antes':>10}{'después':>10}")
    for k in ("brier_global", "ece_global", "brier_fg", "ece_fg",
              "brier_fondo", "ece_fondo"):
        if k in r["antes"]:
            print(f"  {k:<14}{r['antes'][k]:>10.6f}{r['despues'][k]:>10.6f}")

    if args.guardar:
        out = Path(args.resultados) / "calibracion"
        out.mkdir(parents=True, exist_ok=True)
        ruta = out / f"{args.modelo}_s{args.semilla}.json"
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"  guardado: {ruta}")


if __name__ == "__main__":
    main()
