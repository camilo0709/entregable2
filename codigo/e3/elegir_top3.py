"""
Elige los 3 mejores modelos del benchmark POR VALIDACIÓN (nunca por test).

Regla (fijar en protocolo.md antes de entrenar las semillas extra):
  - candidatos: modelos de ORDEN_BENCHMARK entrenables en el driver
    (nnU-Net queda fuera: su validación son folds internos distintos);
  - puntuación: media de dice_val sobre las 5 semillas base;
  - top 3 = mayores medias. Empates: menor desviación estándar.

Salida: <resultados>/top3.json  y  tabla en pantalla.
Uso:    python elegir_top3.py --resultados resultados_e3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelos_tb import MODELOS, ORDEN_BENCHMARK, SEMILLAS  # noqa: E402


def leer_dice(res_dir, modelo, semilla):
    p = os.path.join(res_dir, modelo, f"seed_{semilla}", "resultado.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p, encoding="utf-8"))
    return r["dice_val"] if r.get("estado") == "completado" else None


def elegir(res_dir, k=3):
    cand = [m for m in ORDEN_BENCHMARK
            if MODELOS[m].get("entrenable") and MODELOS[m]["backend"] != "nnunet"]
    filas = []
    for m in cand:
        v = [leer_dice(res_dir, m, s) for s in SEMILLAS]
        v = [x for x in v if x is not None]
        if len(v) == len(SEMILLAS):
            filas.append((m, float(np.mean(v)), float(np.std(v, ddof=1))))
    filas.sort(key=lambda t: (-round(t[1], 6), t[2]))
    faltan = [m for m in cand if m not in [f[0] for f in filas]]
    return filas, faltan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    filas, faltan = elegir(a.resultados, a.k)
    print(f"{'modelo':16s} {'dice_val medio':>14s} {'sd':>8s}")
    for i, (m, mu, sd) in enumerate(filas, 1):
        print(f"{i:2d} {m:13s} {mu:14.4f} {sd:8.4f}" + ("  <- top" if i <= a.k else ""))
    if faltan:
        print("SIN las 5 semillas completas (no se pueden ordenar):", faltan)
        sys.exit("Completa el entrenamiento de las 5 semillas antes de elegir.")
    top = [m for m, _, _ in filas[:a.k]]
    json.dump({"criterio": "media dice_val, 5 semillas base, validación",
               "top": top, "ranking": [{"modelo": m, "dice_val_medio": round(mu, 5),
                                         "sd": round(sd, 5)} for m, mu, sd in filas]},
              open(os.path.join(a.resultados, "top3.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("top3.json escrito:", top)


if __name__ == "__main__":
    main()
