"""
================================================================================
VERIFICADOR DE INTEGRIDAD DE LOS SPLITS SELLADOS
================================================================================

PROPÓSITO:
    Comprueba que los CSV de datos/splits/ siguen siendo EXACTAMENTE los que
    se sellaron en split_manifest.json (SHA-256 + conteos + disjunción de
    ids). Se ejecuta antes de cualquier entrenamiento y después de cualquier
    sincronización (Drive, re-zip, etc.) para detectar corrupción o cambios
    no declarados.

    Los CSV sellados NUNCA se editan; las rutas Windows se remapean al
    cargar (dataset_tb._ruta_local), por lo que los hashes deben seguir
    coincidiendo siempre.

COMPROBACIONES:
    [ ] SHA-256 de train/val/test_pos/test_pos_reserva vs manifest
    [ ] Número de filas por split vs manifest.conteos
    [ ] train ∩ val = ∅ ; (train∪val) ∩ (test_pos∪test_neg) = ∅
    [ ] test_pos ∩ test_neg = ∅ ; sin ids repetidos dentro de cada split
    [ ] columnas mínimas presentes por split

NOTA (R1): los hashes del manifest v1 no corresponden a los CSV recuperados
de Drive (desfase previo a cualquier manipulación local; ver
docs/DECISION_D1_GROUND_TRUTH.md §R1.6). split_manifest_v2.json es la línea
base vigente; v1 se reporta solo como histórico.

USO:
    python codigo/verificar_integridad_splits.py
    python codigo/verificar_integridad_splits.py --manifest otro.json

Código de salida: 0 = todo OK ; 1 = alguna comprobación falla.
================================================================================
"""

import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(ROOT, "datos", "splits")
MANIFEST = os.path.join(SPLITS_DIR, "split_manifest_v2.json")
MANIFEST_V1 = os.path.join(SPLITS_DIR, "split_manifest.json")

COLUMNAS_MIN = {
    "train": {"id", "split", "gt_tipo", "tb_type", "label", "ruta"},
    "val": {"id", "split", "gt_tipo", "tb_type", "label", "ruta"},
    "test_pos": {"id", "gt_tipo", "n_boxes", "ruta"},
    "test_neg": {"id", "gt_tipo", "ruta"},
    "test_pos_reserva": {"id", "gt_tipo", "n_boxes", "ruta"},
}


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_ids(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return filas, [r["id"] for r in filas]


def main():
    ap = argparse.ArgumentParser(description="Verifica los splits sellados")
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()

    fallos = []

    print("=" * 74)
    print("VERIFICACIÓN DE INTEGRIDAD — SPLITS SELLADOS")
    print("=" * 74)

    if not os.path.exists(a.manifest):
        print(f"  [!!] No existe el manifest: {a.manifest}")
        return 1
    with open(a.manifest, encoding="utf-8") as f:
        man = json.load(f)

    # ---------------------------------------------------------- hashes
    print("\n-- SHA-256 vs línea base vigente (manifest v2) --")
    for sp, esperado in man.get("hash_integridad_v2", {}).items():
        ruta = os.path.join(SPLITS_DIR, f"{sp}.csv")
        if not os.path.exists(ruta):
            fallos.append(f"{sp}.csv ausente")
            print(f"  [!!] {sp:18s} ARCHIVO AUSENTE: {ruta}")
            continue
        actual = sha256(ruta)
        ok = actual == esperado
        print(f"  [{'OK ' if ok else '!!!'}] {sp:18s} {actual[:16]}…")
        if not ok:
            fallos.append(f"{sp}.csv hash != manifest v2 "
                          f"({actual[:12]} != {esperado[:12]})")

    # ----------------------------------------------- histórico (v1)
    if os.path.exists(MANIFEST_V1):
        try:
            with open(MANIFEST_V1, encoding="utf-8") as f:
                v1 = json.load(f)
            dif = []
            for sp, esperado in v1.get("hash_integridad", {}).items():
                ruta = os.path.join(SPLITS_DIR, f"{sp}.csv")
                if os.path.exists(ruta) and sha256(ruta) != esperado:
                    dif.append(sp)
            if dif:
                print(f"\n  (histórico) manifest v1: {len(dif)} hashes no "
                      f"verificables — ya documentado en D1-R1 §R1.6, no es "
                      f"fallo nuevo")
        except Exception:
            pass

    # --------------------------------------------------------- conteos
    print("\n-- Contenidos --")
    ids = {}
    for sp in COLUMNAS_MIN:
        ruta = os.path.join(SPLITS_DIR, f"{sp}.csv")
        if not os.path.exists(ruta):
            continue
        filas, l = leer_ids(ruta)
        ids[sp] = l
        esperado = man.get("conteos", {}).get(sp)
        if sp == "test_pos_reserva":
            print(f"  [--] {sp:18s} n={len(l)} (sin conteo en manifest)")
        else:
            ok = esperado == len(l)
            print(f"  [{'OK ' if ok else '!!!'}] {sp:18s} n={len(l)}"
                  + ("" if ok else f"  (manifest dice {esperado})"))
            if not ok:
                fallos.append(f"{sp}: {len(l)} filas != manifest {esperado}")
        faltan_cols = COLUMNAS_MIN[sp] - set(filas[0].keys() if filas else [])
        if faltan_cols:
            fallos.append(f"{sp}: faltan columnas {sorted(faltan_cols)}")
            print(f"  [!!!] {sp:18s} faltan columnas: {sorted(faltan_cols)}")
        repetidos = len(l) - len(set(l))
        if repetidos:
            fallos.append(f"{sp}: {repetidos} ids repetidos")
            print(f"  [!!!] {sp:18s} {repetidos} ids repetidos")

    # ------------------------------------------------------ disjunción
    print("\n-- Disjunción entre splits --")
    pares = [("train", "val"),
             ("train", "test_pos"), ("val", "test_pos"),
             ("train", "test_neg"), ("val", "test_neg"),
             ("test_pos", "test_neg")]
    for x, y in pares:
        if x in ids and y in ids:
            inter = set(ids[x]) & set(ids[y])
            print(f"  [{'OK ' if not inter else '!!!'}] {x} ∩ {y} = {len(inter)}")
            if inter:
                fallos.append(f"{x} ∩ {y} no vacío: {sorted(inter)[:5]}")

    # --------------------------------------------------------- resumen
    print("\n" + "-" * 74)
    if not fallos:
        print("SPLITS ÍNTEGROS — todos los sellados verificados.")
        return 0
    print(f"FALLA: {len(fallos)} problema(s):")
    for x in fallos:
        print(f"   - {x}")
    print("NO entrenar hasta resolver esto.")
    return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
