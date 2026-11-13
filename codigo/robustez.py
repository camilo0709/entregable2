"""
robustez.py — Robustez ante perturbaciones de la imagen de entrada.

QUÉ HACE
    Aplica perturbaciones controladas a la imagen original (antes del z-score
    por imagen), infiere con el checkpoint congelado y mide la caída de Dice,
    IoU, HD95 y ASSD respecto de la condición limpia.

POR QUÉ ANTES DEL Z-SCORE
    El modelo se entrenó con z-score por imagen. Perturbar después del
    z-score cambiaría la escala de forma no equivalente a una degradación
    real de la radiografía. Aquí se perturba en el espacio de intensidades
    0–255 y se vuelve a normalizar, que es lo que ocurriría con una imagen
    capturada en peores condiciones.

PERTURBACIONES (declaradas, no optimizadas sobre test)
    - ruido gaussiano aditivo σ ∈ {5, 10, 20} (niveles de uint8)
    - desenfoque gaussiano σ ∈ {1, 2} píxeles
    - gamma ∈ {0,8; 1,25} (contraste global)
    - rotación ±10° (geométrica, sin tocar la máscara)
    - limpio (referencia)

USO
    python codigo/robustez.py --modelo deeplabv3p --semilla 42 --limite 4
    python codigo/robustez.py --modelo deeplabv3p --semilla 42
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_tb import DatasetTB, _normalizar, _ruta_local   # noqa: E402
from evaluar_test import cargar_modelo                         # noqa: E402
from metricas_tb import metricas_imagen                        # noqa: E402

UMBRAL = 0.5


# --------------------------------------------------------------------------
# Perturbaciones en espacio 0–255
# --------------------------------------------------------------------------
def _clip(img):
    return np.clip(img, 0.0, 255.0).astype(np.float32)


def p_limpio(img, rng):
    return img.astype(np.float32)


def p_ruido(sigma):
    def f(img, rng):
        return _clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape))
    return f


def p_blur(sigma):
    def f(img, rng):
        return ndimage.gaussian_filter(img.astype(np.float32), sigma=sigma)
    return f


def p_gamma(gamma):
    def f(img, rng):
        x = np.clip(img.astype(np.float32) / 255.0, 0, 1)
        return (255.0 * np.power(x, gamma)).astype(np.float32)
    return f


def p_rotacion(angulo):
    def f(img, rng):
        return ndimage.rotate(img.astype(np.float32), angulo, reshape=False,
                              order=1, mode="nearest")
    return f


CONDICIONES = {
    "limpio": p_limpio,
    "ruido_5": p_ruido(5),
    "ruido_10": p_ruido(10),
    "ruido_20": p_ruido(20),
    "blur_1": p_blur(1.0),
    "blur_2": p_blur(2.0),
    "gamma_0.8": p_gamma(0.8),
    "gamma_1.25": p_gamma(1.25),
    "rot_m10": p_rotacion(-10.0),
    "rot_p10": p_rotacion(10.0),
}


# --------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------
def _logits(net, x):
    salida = net(x)
    if isinstance(salida, (tuple, list)):
        for s in salida:
            if isinstance(s, torch.Tensor) and s.ndim == 4:
                return s
        raise TypeError("salida del modelo sin tensor 4D")
    return salida


def _cargar_imagen(ds, fila):
    """Imagen original (0–255) del split, sin normalizar."""
    if ds.split in ("train", "val"):
        return ds._imagen_desde_npz(_ruta_local(fila["ruta"]))
    return ds._imagen_desde_png(_ruta_local(fila["ruta"]))


def _cargar_gt(ds, fila):
    if ds.split in ("train", "val"):
        return ds._cargar_mask(_ruta_local(fila["ruta"])).astype(bool)
    return ds._cargar_mask(fila).astype(bool)


@torch.no_grad()
def evaluar_condicion(net, ds, condicion, fn, device, rng, batch=8):
    """Devuelve una lista de dicts de métricas para una condición."""
    filas = []
    buffer_img, buffer_gt, buffer_id = [], [], []

    def volcar():
        if not buffer_img:
            return
        x = torch.from_numpy(np.stack(buffer_img)[:, None]).to(device)
        prob = torch.sigmoid(_logits(net, x))[:, 0].cpu().numpy()
        for i, pid in enumerate(buffer_id):
            pred = prob[i] > UMBRAL
            m = metricas_imagen(pred, buffer_gt[i])
            m["id"] = pid
            m["condicion"] = condicion
            filas.append(m)
        buffer_img.clear()
        buffer_gt.clear()
        buffer_id.clear()

    for i, fila in enumerate(ds.filas):
        raw = _cargar_imagen(ds, fila)
        gt = _cargar_gt(ds, fila)
        pert = fn(raw, rng)
        if pert.shape != raw.shape:
            raise ValueError(f"perturbación cambió la forma en {fila['id']}")
        buffer_img.append(_normalizar(pert))
        buffer_gt.append(gt)
        buffer_id.append(fila["id"])
        if len(buffer_img) >= batch:
            volcar()
    volcar()
    return filas


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Robustez ante perturbaciones")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--semilla", type=int, required=True)
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--split", default="test_pos", choices=["val", "test_pos"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--condiciones", default="todas",
                    help="lista separada por comas o 'todas'")
    ap.add_argument("--semilla-ruido", type=int, default=2026)
    args = ap.parse_args()

    rd = os.path.join(args.resultados, args.modelo, f"seed_{args.semilla}")
    ckpt = os.path.join(rd, "mejor_modelo.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"sin checkpoint: {ckpt}")

    if args.condiciones == "todas":
        nombres = list(CONDICIONES)
    else:
        nombres = [c.strip() for c in args.condiciones.split(",") if c.strip()]
        faltan = [c for c in nombres if c not in CONDICIONES]
        if faltan:
            raise SystemExit(f"condiciones desconocidas: {faltan}")

    net = cargar_modelo(args.modelo, ckpt, torch.device(args.device))
    ds = DatasetTB(args.split, esperar_gt=True, verbose=False)
    if args.limite is not None:
        ds.filas = ds.filas[:args.limite]

    rng = np.random.default_rng(args.semilla_ruido)
    todas = []
    for nombre in nombres:
        print(f"  {nombre} ...", flush=True)
        todas.extend(evaluar_condicion(net, ds, nombre, CONDICIONES[nombre],
                                       torch.device(args.device), rng))

    out = Path(args.resultados) / "robustez"
    out.mkdir(parents=True, exist_ok=True)
    ruta = out / f"{args.modelo}_s{args.semilla}.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)

    base = [r["dice"] for r in todas if r["condicion"] == "limpio"]
    base_mean = float(np.mean(base)) if base else float("nan")
    print(f"\n{args.modelo} · semilla {args.semilla} · {args.split} "
          f"({len(ds)} imágenes)")
    print(f"  {'condición':<12}{'Dice':>9}{'ΔDice':>9}{'IoU':>9}")
    for nombre in nombres:
        d = [r["dice"] for r in todas if r["condicion"] == nombre]
        io = [r["iou"] for r in todas if r["condicion"] == nombre]
        print(f"  {nombre:<12}{np.mean(d):>9.4f}{np.mean(d) - base_mean:>+9.4f}"
              f"{np.mean(io):>9.4f}")
    print(f"  guardado: {ruta}")


if __name__ == "__main__":
    main()
