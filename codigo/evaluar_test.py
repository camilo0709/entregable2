"""
evaluar_test.py — Fase 6: evaluación de la prueba reservada (test_pos).

Garantía estructural (Sección 4.1.2): el driver de entrenamiento NUNCA abre
test. Este script se ejecuta una sola vez, cuando todas las corridas están
congeladas, y para cada corrida COMPLETADA:

  1. carga el checkpoint de la MEJOR época (seleccionada por validación),
  2. infiere sobre las imágenes de test_pos (GT plata D1-R1 congelado),
  3. aplica el umbral fijo 0.5 (preregistro: sin calibración por modelo),
  4. escribe `por_imagen_test.csv` y actualiza `resultado.json`
     (campo `dice_test` = media del Dice por imagen del test).

Opcional (--ood): control OOD sobre test_neg (radiografías sanas, máscara
vacía por construcción): se registra la fracción de píxeles predichos como
lesión (debe ser ~0) en `ood_test.csv` y en `resultado.json["ood"]`.

Reanudable: omite las corridas que ya tengan `dice_test` (salvo --forzar).

Uso (Colab, GPU):
    python codigo/evaluar_test.py --resultados <RESULTADOS> --ood
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driver_entrenamiento import (recomponer_resumen, ruta_run,  # noqa: E402
                                  run_completado)
from dataset_tb import DatasetTB                            # noqa: E402
from metricas_tb import metricas_imagen                     # noqa: E402
from modelos_tb import (MODELOS, ORDEN_BASELINES,           # noqa: E402
                        ORDEN_BENCHMARK, SEMILLAS, crear_modelo)

UMBRAL = 0.5   # preregistro: fijo para todos los modelos, sin calibración


def cargar_modelo(clave: str, ruta_ckpt: str, device) -> torch.nn.Module:
    """Construye la arquitectura y carga el checkpoint de la mejor época."""
    net, _ = crear_modelo(clave, in_channels=1, out_channels=1, verbose=False)
    ckpt = torch.load(ruta_ckpt, map_location="cpu")
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def inferir(net, ds, device, batch_size=8, umbral=UMBRAL, workers=0):
    """Devuelve (ids, lista de predicciones booleanas 512x512)."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers)
    ids, preds = [], []
    for imgs, _, infos in dl:
        logits = net(imgs.to(device))
        prob = torch.sigmoid(logits)[:, 0]           # (B, H, W)
        m = (prob > umbral).cpu().numpy().astype(bool)
        preds.extend(m)
        ids.extend(infos["id"])
    return ids, preds


def evaluar_corrida(clave, semilla, res_dir, splits_dir, device, args):
    """Evalúa UNA corrida sobre test_pos (+ opcional test_neg)."""
    rd = ruta_run(res_dir, clave, semilla)
    ruta_json = os.path.join(rd, "resultado.json")
    ruta_ckpt = os.path.join(rd, "mejor_modelo.pt")
    if not os.path.exists(ruta_ckpt):
        raise FileNotFoundError(f"sin checkpoint: {ruta_ckpt}")

    with open(ruta_json, encoding="utf-8") as f:
        r = json.load(f)

    if "dice_test" in r and not args.forzar:
        return "omitida", r["dice_test"]

    net = cargar_modelo(clave, ruta_ckpt, device)

    ds_te = DatasetTB("test_pos", splits_dir=splits_dir, esperar_gt=True,
                      verbose=False)
    ids, preds = inferir(net, ds_te, device, args.batch, args.umbral,
                         args.workers)
    assert len(preds) == len(ds_te), "conteo de predicciones != test"

    filas = []
    for i, pred in enumerate(preds):
        fila_gt = ds_te.filas[i]
        gt = ds_te._cargar_mask(fila_gt).astype(bool)
        m = metricas_imagen(pred, gt)
        m["id"] = ids[i]
        filas.append(m)
    with open(os.path.join(rd, "por_imagen_test.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    dice_test = round(float(np.mean([x["dice"] for x in filas])), 5)

    # ------------------------------------------- control OOD (opcional)
    ood = None
    if args.ood:
        ds_ood = DatasetTB("test_neg", splits_dir=splits_dir,
                           esperar_gt=False, verbose=False)
        ids_o, preds_o = inferir(net, ds_ood, device, args.batch, args.umbral,
                                 args.workers)
        fracs = [float(p.mean()) for p in preds_o]
        ood = {"n": len(fracs),
               "frac_media": round(float(np.mean(fracs)), 6),
               "frac_max": round(float(np.max(fracs)), 6),
               "n_con_prediccion": int(sum(p.any() for p in preds_o))}
        with open(os.path.join(rd, "ood_test.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "frac_predichos_lesion"])
            for i, fr in enumerate(fracs):
                w.writerow([ids_o[i], round(fr, 6)])

    r["dice_test"] = dice_test
    r["n_test"] = len(filas)
    r["umbral"] = args.umbral
    r["gt_test"] = "plata SAM2 congelado (D1-R1)"
    r["test_evaluado_en"] = datetime.now().isoformat(timespec="seconds")
    if ood is not None:
        r["ood"] = ood
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    return "evaluada", dice_test


def main():
    ap = argparse.ArgumentParser(
        description="Fase 6: evaluación de test (una vez, corridas congeladas)")
    ap.add_argument("--resultados", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--modelos", nargs="*", default=None)
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--ood", action="store_true",
                    help="Suma el control OOD sobre test_neg")
    ap.add_argument("--forzar", action="store_true",
                    help="Re-evalúa aunque ya exista dice_test")
    a = ap.parse_args()

    raiz = Path(__file__).resolve().parents[1]
    res_dir = a.resultados or os.path.join(raiz, "resultados")
    splits_dir = a.splits_dir or os.path.join(raiz, "datos", "splits")

    entrenables = [m for m in ORDEN_BENCHMARK + ORDEN_BASELINES
                   if MODELOS.get(m, {}).get("entrenable", False)]
    modelos = a.modelos or entrenables
    semillas = a.semillas or SEMILLAS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print("FASE 6 - EVALUACION DE LA PRUEBA RESERVADA")
    print("=" * 74)
    print(f"Device   : {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    print(f"Modelos  : {len(modelos)} | Semillas: {semillas}")
    print(f"Test     : test_pos (GT plata D1-R1) umbral={a.umbral}"
          + (" + control OOD test_neg" if a.ood else ""))

    t0 = time.time()
    hechas, omitidas, errores, pendientes_sin_run = 0, 0, [], []
    for m in modelos:
        for s in semillas:
            if not run_completado(res_dir, m, s):
                pendientes_sin_run.append((m, s))
                continue
            etiqueta = f"{m:15s} semilla {s:<6}"
            try:
                estado, dice = evaluar_corrida(m, s, res_dir, splits_dir,
                                               device, a)
                if estado == "omitida":
                    omitidas += 1
                    print(f"  = {etiqueta} ya evaluada (dice_test={dice})")
                else:
                    hechas += 1
                    print(f"  ✓ {etiqueta} dice_test={dice}")
            except Exception as e:
                errores.append((m, s, f"{type(e).__name__}: {e}"))
                print(f"  ✗ {etiqueta} ERROR: {type(e).__name__}: {e}")

    if pendientes_sin_run:
        print(f"\nSin corrida completada (aún sin entrenar): "
              f"{len(pendientes_sin_run)} -> {pendientes_sin_run[:5]}"
              + ("..." if len(pendientes_sin_run) > 5 else ""))
    print("-" * 74)
    print(f"Evaluadas: {hechas} | Ya estaban: {omitidas} | "
          f"Errores: {len(errores)} | {time.time() - t0:.0f} s")
    for m, s, e in errores:
        print(f"   {m} s{s}: {e}")

    filas = recomponer_resumen(res_dir, modelos, semillas)
    print(f"resumen_runs.csv regenerado con {len(filas)} filas")
    print("\nSiguiente: codigo/estadistica_tb.py (Tablas A/B/C).")


if __name__ == "__main__":
    main()
