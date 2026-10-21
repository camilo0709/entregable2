"""
================================================================================
VALIDAR GROUND TRUTH DEL TEST (D1-R1) — CONGELADO DEL GT
================================================================================

PROPÓSITO:
    Ejecuta TODAS las comprobaciones de calidad del ground truth
    antes de congelarlo (docs/PROTOCOLO_ANOTACION.md §6 y §9):
    aplica por igual al GT plata SAM 2 (D1-R1) y a las máscaras
    humanas de la submuestra de validación (--anotador-b).

      [ ] Existen exactamente los 80 PNG de test_pos.csv (sin huecos)
      [ ] Cada PNG mide 512x512
      [ ] Valores ⊆ {0, 255} (binario puro, sin grises)
      [ ] Ninguna máscara vacía      (todo test_pos es positivo)
      [ ] Áreas en rango razonable   (< 100 px o > 40 % => revisar)
      [ ] Sin archivos huérfanos     (PNG que no están en el split)
      [ ] Dice inter-anotador ≥ 0,70 (si se pasa --anotador-b)

    Si todo pasa, escribe ground_truth_test/ANOTACION_METRICAS.json y el GT
    queda listo para congelar. Si algo falla, devuelve código de salida 1
    para que nada posterior (entrenamiento/evaluación) continúe sobre un GT
    defectuoso.

USO:
    python codigo/validar_ground_truth_test.py
    python codigo/validar_ground_truth_test.py --anotador-b ruta/al/anotador_b
    python codigo/validar_ground_truth_test.py --solo-formato   (rápido)
================================================================================
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metricas_tb import dice                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(ROOT, "datos", "splits")
TEST_POS = os.path.join(SPLITS_DIR, "test_pos.csv")
GT_DIR = os.path.join(ROOT, "datos", "ground_truth_test")

ESPERADO_SIZE = (512, 512)
MIN_PIXELES = 100            # por debajo => marcar para revisión
MAX_FRACCION = 0.40          # por encima => marcar para revisión
UMBRAL_DICE_INTER = 0.70     # criterio de aceptación §4.1
N_DOBLE = 20
SEMILLA_ORDEN = 42           # debe coincidir con el exportador


# ============================================================================

def cargar_gt(ruta):
    """Carga un PNG de GT como bool y devuelve (mascara, problemas_estructura)."""
    problemas = []
    with Image.open(ruta) as im:
        size = im.size
        arr = np.array(im)
    if size != ESPERADO_SIZE:
        problemas.append(f"tamaño {size}, esperado {ESPERADO_SIZE}")
    if arr.ndim == 3:                       # si tiene canal alfa/RGB, colapsar
        arr = arr[..., 0]
    valores = np.unique(arr)
    if not np.isin(valores, [0, 255]).all():
        problemas.append(f"valores distintos de {{0,255}}: "
                         f"{valores[:10].tolist()}{'...' if valores.size > 10 else ''}")
    return (arr > 127), problemas


def main():
    ap = argparse.ArgumentParser(
        description="Valida el GT del test (plata D1-R1) antes de congelarlo.")
    ap.add_argument("--gt-dir", default=GT_DIR,
                    help="Directorio con los PNG anotados")
    ap.add_argument("--anotador-b", default=None,
                    help="Directorio con la doble anotación (submuestra) "
                         "para calcular Dice inter-anotador")
    ap.add_argument("--solo-formato", action="store_true",
                    help="Solo chequeos de formato/contenido (omite áreas e "
                         "inter-anotador)")
    a = ap.parse_args()

    # La consola de Windows (cp1252) no puede imprimir ≥ / ✗ y abortaría el
    # informe a mitad de camino; se fuerza UTF-8 con reemplazo seguro.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 74)
    print("VALIDACIÓN DE GROUND TRUTH MANUAL — TEST POSITIVO (D1)")
    print("=" * 74)

    if not os.path.exists(TEST_POS):
        raise SystemExit(f"No existe {TEST_POS}")
    with open(TEST_POS, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    ids = [r["id"] for r in filas]
    n_esp = len(ids)

    fallos, revisiones = [], []
    informe = {"validado": datetime.now().isoformat(timespec="seconds"),
               "gt_dir": a.gt_dir, "n_esperado": n_esp}

    # ------------------------------------------------------ 1. cobertura
    if not os.path.isdir(a.gt_dir):
        print(f"  [FALLO] No existe el directorio de GT: {a.gt_dir}")
        raise SystemExit(1)

    presentes = sorted(fn[:-4] for fn in os.listdir(a.gt_dir)
                       if fn.lower().endswith(".png"))
    faltantes = [i for i in ids if i not in presentes]
    huerfanos = [i for i in presentes if i not in ids]
    print(f"  [{'OK ' if not faltantes else 'FALLO'}] Cobertura: "
          f"{n_esp - len(faltantes)}/{n_esp} presentes, "
          f"{len(faltantes)} faltantes, {len(huerfanos)} huérfanos")
    if faltantes:
        fallos.append(f"faltan {len(faltantes)} PNG: {faltantes[:10]}")
    if huerfanos:
        fallos.append(f"archivos huérfanos (no están en test_pos): "
                      f"{huerfanos[:10]}")

    # ------------------------------------------------- 2-4. por imagen
    n_vacias, n_mal_formato = 0, 0
    areas = {}
    for iid in ids:
        ruta = os.path.join(a.gt_dir, f"{iid}.png")
        if not os.path.exists(ruta):
            continue
        m, probs = cargar_gt(ruta)
        if probs:
            n_mal_formato += 1
            fallos.append(f"{iid}: {'; '.join(probs)}")
        if not m.any():
            n_vacias += 1
            fallos.append(f"{iid}: máscara vacía (test_pos es positivo)")
        areas[iid] = int(m.sum())

    print(f"  [{'OK ' if n_mal_formato == 0 else 'FALLO'}] Formato binario "
          f"512x512: {n_mal_formato} con problemas")
    print(f"  [{'OK ' if n_vacias == 0 else 'FALLO'}] Máscaras vacías: {n_vacias}")

    # ------------------------------------------------------ 5. áreas
    if not a.solo_formato and areas:
        bajos = [k for k, v in areas.items() if v < MIN_PIXELES]
        altos = [k for k, v in areas.items()
                 if v > MAX_FRACCION * 512 * 512]
        estado = "OK " if not bajos and not altos else "REV"
        print(f"  [{estado}] Áreas: min={min(areas.values())} "
              f"max={max(areas.values())} "
              f"(umbral {MIN_PIXELES} px … {int(MAX_FRACCION*512*512)} px)")
        if bajos:
            revisiones.append(f"área < {MIN_PIXELES} px: {bajos}")
        if altos:
            revisiones.append(f"área > {int(MAX_FRACCION*100)} % de la imagen: {altos}")
        for r in revisiones:
            print(f"        revisar -> {r}")
        informe["areas_px"] = {k: areas[k] for k in sorted(areas)}

    # ------------------------------------------- 6. Dice inter-anotador
    if a.anotador_b:
        # La submuestra la fija el exportador (§9, estratificada semilla 42):
        # se leen sus ids desde submuestra.csv para no reproducir con otro
        # algoritmo una selección distinta. (Fallback legado: shuffle numpy.)
        ruta_sub = os.path.join(os.path.dirname(os.path.normpath(a.anotador_b)),
                                "submuestra.csv")
        if os.path.exists(ruta_sub):
            with open(ruta_sub, newline="", encoding="utf-8") as f:
                submuestra = [r["id"] for r in csv.DictReader(f)]
        else:
            orden = list(ids)
            np.random.RandomState(SEMILLA_ORDEN).shuffle(orden)
            submuestra = orden[:N_DOBLE]
        pares = []
        for iid in submuestra:
            ra = os.path.join(a.gt_dir, f"{iid}.png")
            rb = os.path.join(a.anotador_b, f"{iid}.png")
            if not (os.path.exists(ra) and os.path.exists(rb)):
                fallos.append(f"{iid}: falta en submuestra doble "
                              f"(A={'sí' if os.path.exists(ra) else 'no'}, "
                              f"B={'sí' if os.path.exists(rb) else 'no'})")
                continue
            ma, _ = cargar_gt(ra)
            mb, _ = cargar_gt(rb)
            pares.append((iid, dice(ma, mb)))
        if pares:
            d = np.array([v for _, v in pares])
            media = float(d.mean())
            informe["dice_inter_anotador"] = {
                "n": len(pares), "mean": round(media, 4),
                "std": round(float(d.std()), 4),
                "min": round(float(d.min()), 4),
                "pares": {k: round(v, 4) for k, v in pares},
                "umbral": UMBRAL_DICE_INTER,
            }
            ok = media >= UMBRAL_DICE_INTER
            print(f"  [{'OK ' if ok else 'FALLO'}] Dice inter-anotador: "
                  f"media={media:.4f} (mín {d.min():.4f}) sobre {len(pares)} "
                  f"imágenes [umbral ≥ {UMBRAL_DICE_INTER}]")
            if not ok:
                fallos.append(
                    f"Dice inter-anotador {media:.4f} < {UMBRAL_DICE_INTER}: "
                    "reescribir reglas §2 y reanotar la submuestra (§4.1)")
            for iid, v in sorted(pares, key=lambda x: x[1]):
                if v < 0.60:
                    print(f"        consenso requerido -> {iid}: {v:.4f}")
    else:
        print("  [ -- ] Dice inter-anotador: omitido (pasar --anotador-b)")

    # ---------------------------------------------------------- veredicto
    informe["fallos"] = fallos
    informe["revisiones"] = revisiones
    informe["apto"] = not fallos
    print("-" * 74)

    if fallos:
        print(f"GT NO APTO: {len(fallos)} fallo(s)")
        for x in fallos[:20]:
            print(f"   ✗ {x}")
        if len(fallos) > 20:
            print(f"   … y {len(fallos) - 20} más")
        raise SystemExit(1)

    print("GT APTO. Todas las comprobaciones pasaron.")
    if revisiones:
        print("Nota: hay imágenes marcadas para revisión manual "
              "(no bloquean, pero se declaran).")

    # ------------------------------------------------------------ guardar
    ruta_json = os.path.join(a.gt_dir, "ANOTACION_METRICAS.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
    print(f"Métricas de calidad: {ruta_json}")
    print("Siguiente paso: congelar el GT (§5.3) y no modificarlo más "
          "sin registrar el cambio en ANOTACION_CAMBIOS.md.")


if __name__ == "__main__":
    main()
