"""
Limpieza y auditoría de resultados/resumen_runs.csv  (Entregable 3, Bloque A)

Hace, de forma reproducible (sin editar a mano):
  1. elimina filas duplicadas exactas (p. ej. nnunet repetido 6 veces);
  2. separa corridas completadas de las no disponibles;
  3. detecta modelos con resultados idénticos en TODAS las semillas
     (señal de que son la misma red: bug de configuración o falta de HPO);
  4. cuenta modelos y corridas para reconciliar las cifras del libro.

Uso:  python limpiar_resultados.py resultados/resumen_runs.csv
Salidas (junto al CSV): resumen_runs_limpio.csv y auditoria_resultados.md
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


def auditar(ruta):
    ruta = Path(ruta)
    crudo = pd.read_csv(ruta)
    limpio = crudo.drop_duplicates().reset_index(drop=True)
    ok = limpio[limpio.estado == "completado"]
    no = limpio[limpio.estado != "completado"]

    pivote = ok.pivot(index="semilla", columns="modelo", values="dice_val")
    idem = [(a, b) for a, b in combinations(pivote.columns, 2)
            if (pivote[a].round(5) == pivote[b].round(5)).all()]

    lineas = [
        "# Auditoría de resumen_runs.csv", "",
        f"- Filas crudas: {len(crudo)}; tras quitar duplicados exactos: "
        f"{len(limpio)} ({len(crudo) - len(limpio)} eliminadas).",
        f"- Corridas completadas: {len(ok)} de "
        f"{ok.modelo.nunique()} modelos "
        f"({ok.groupby('modelo').size().min()}-"
        f"{ok.groupby('modelo').size().max()} semillas cada uno).",
        f"- No disponibles: {len(no)} filas de "
        f"{sorted(no.modelo.unique())}.", "",
        "## Pares con Dice idéntico en todas las semillas", "",
    ]
    lineas += ([f"- {a} == {b}" for a, b in idem] or ["- ninguno"])
    lineas += ["", "## Dice de validación por modelo (media ± sd)", "",
               "| modelo | n | media | sd |", "|---|---|---|---|"]
    for m, g in ok.groupby("modelo"):
        lineas.append(f"| {m} | {len(g)} | {g.dice_val.mean():.4f} | "
                      f"{g.dice_val.std():.4f} |")
    limpio.to_csv(ruta.with_name("resumen_runs_limpio.csv"), index=False)
    ruta.with_name("auditoria_resultados.md").write_text(
        "\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    return limpio, idem


if __name__ == "__main__":
    auditar(sys.argv[1] if len(sys.argv) > 1 else "resultados/resumen_runs.csv")
