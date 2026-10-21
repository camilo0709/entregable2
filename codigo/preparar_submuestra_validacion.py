"""
================================================================================
PREPARAR SUBMUESTRA DE VALIDACIÓN HUMANA (AJUSTE R1)
================================================================================

PROPÓSITO:
    Selecciona la submuestra aleatoria ESTRATIFICADA (semilla 42) de
    imágenes de test_pos que el usuario validará/corregirá a mano, según
    docs/PROTOCOLO_ANOTACION.md §9 (ajuste D1-R1). El Dice resultante
    entre la máscara humana y la máscara SAM 2 se reporta como TECHO DE
    DESEMPEÑO del benchmark.

    Estratificación por n_boxes (focos): se garantiza al menos 1 imagen
    por estrato presente y el resto proporcional al tamaño del estrato.

SALIDA (datos/submuestra_validacion/):
    submuestra.csv        id, n_boxes, estrato y rutas de cada elemento
    imagenes/<id>.png     copia de la radiografía (512x512)
    mascaras_sam2/<id>.png  GT plata SAM 2 (lo que el validador corrige)
    overlays/<id>.png     imagen + máscara translúcida + contorno + caja(s)
                          naranja(s) del anotador original (orientación)

EL VALIDADOR:
    Revisa cada overlay; si la máscara verde no refleja la lesión, corrige
    con cualquier editor de pincel y exporta PNG binario 0/255 512x512 en:
        datos/submuestra_validacion/mascaras_humano/<id>.png
    Si aprueba la máscara tal cual, copia el PNG SAM 2 sin modificar.

USO:
    python codigo/preparar_submuestra_validacion.py                 # 20 imgs
    python codigo/preparar_submuestra_validacion.py --n 15 --semilla 42
================================================================================
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(ROOT, "datos", "splits")
TBX_TB = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K", "imgs", "tb")
TBX_JSON = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K",
                        "annotations", "json")
GT_DIR = os.path.join(ROOT, "datos", "ground_truth_test")
OUT_DIR = os.path.join(ROOT, "datos", "submuestra_validacion")

SEMILLA = 42
N_DEFAULT = 20


def cajas_por_id():
    """{id: [(x,y,w,h), ...]} desde los JSON COCO de TBX11K (solo stdlib)."""
    mapa = {}
    for nombre in ("TBX11K_train_only_tb.json", "TBX11K_val.json"):
        ruta = os.path.join(TBX_JSON, nombre)
        if not os.path.exists(ruta):
            continue
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        imgs = {im["id"]: im for im in d.get("images", [])}
        for a in d.get("annotations", []):
            im = imgs.get(a["image_id"])
            if im is None:
                continue
            clave = os.path.splitext(os.path.basename(im["file_name"]))[0]
            mapa.setdefault(clave, []).append(tuple(a["bbox"]))
    return mapa


def estratificar(filas, n, semilla):
    """Muestra estratificada por n_boxes: >=1 por estrato + proporcional."""
    grupos = defaultdict(list)
    for fila in filas:
        grupos[int(fila.get("n_boxes", 1) or 1)].append(fila)
    for g in grupos.values():
        g.sort(key=lambda r: r["id"])          # orden estable previo al rng
    rng = random.Random(semilla)

    elegidas = []
    claves = sorted(grupos)
    # 1) al menos una por estrato (si n lo permite)
    for c in claves:
        if len(elegidas) < n:
            elegidas.append(rng.choice(grupos[c]))
    # 2) el resto proporcional a tamaño de estrato
    restantes = []
    for c in claves:
        ya = {f["id"] for f in elegidas}
        pool = [f for f in grupos[c] if f["id"] not in ya]
        rng.shuffle(pool)
        restantes.extend(pool)
    rng.shuffle(restantes)
    elegidas.extend(restantes[: max(0, n - len(elegidas))])

    def estrato_de(f):
        c = int(f.get("n_boxes", 1) or 1)
        return f"{c}_box" if c > 1 else "1_box"
    return elegidas, {f["id"]: estrato_de(f) for f in elegidas}


def overlay(imagen_u8, mask_u8, cajas, salida):
    """Imagen + máscara verde translúcida + contorno verde 1px + cajas naranjas."""
    base = np.array(Image.fromarray(imagen_u8).convert("RGBA"))
    m = mask_u8 > 127

    # contorno = máscara − erosión 1 px (vecindario 4)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    anillo = m & ~er

    out = base.copy()
    a = 110.0 / 255.0                      # translucidez original
    out[m, :3] = (base[m, :3] * (1 - a)
                  + np.array([0, 255, 0], dtype=np.float32) * a
                  ).astype(np.uint8)
    out[anillo, :3] = (0, 255, 0)          # contorno sólido

    img = Image.fromarray(out, "RGBA").convert("RGB")
    d = ImageDraw.Draw(img)
    for (x, y, w, h) in cajas:
        d.rectangle([x, y, x + w, y + h], outline=(255, 140, 0), width=2)
    img.save(salida)


def main():
    ap = argparse.ArgumentParser(
        description="Prepara la submuestra de validación humana (R1)")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 74)
    print("SUBMUESTRA DE VALIDACIÓN HUMANA (PROTOCOLO R1 §9)")
    print("=" * 74)

    if not os.path.isdir(GT_DIR):
        raise SystemExit(
            f"Falta el GT plata: {GT_DIR}\n"
            "  Primero: python codigo/generar_pseudo_gt.py --splits test_pos")

    with open(os.path.join(SPLITS_DIR, "test_pos.csv"),
              newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    print(f"  test_pos: {len(filas)} imágenes | n={a.n} | semilla={a.semilla}")

    elegidas, estratos = estratificar(filas, a.n, a.semilla)
    por_id = {f["id"]: f for f in filas}
    cajas = cajas_por_id()

    for sub in ("imagenes", "mascaras_sam2", "overlays", "mascaras_humano"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    import numpy as np
    faltantes = []
    with open(os.path.join(a.out, "submuestra.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "n_boxes", "estrato", "ruta_imagen",
                    "ruta_mask_sam2", "ruta_overlay"])
        for fila in elegidas:
            iid = fila["id"]
            ruta_img = os.path.join(TBX_TB, f"{iid}.png")
            ruta_ms2 = os.path.join(GT_DIR, f"{iid}.png")
            if not (os.path.exists(ruta_img) and os.path.exists(ruta_ms2)):
                faltantes.append(iid)
                continue
            img = np.array(Image.open(ruta_img).convert("L"))
            msk = np.array(Image.open(ruta_ms2).convert("L"))
            Image.fromarray(img).save(os.path.join(a.out, "imagenes",
                                                   f"{iid}.png"))
            Image.fromarray(msk).save(os.path.join(a.out, "mascaras_sam2",
                                                   f"{iid}.png"))
            ov = os.path.join(a.out, "overlays", f"{iid}.png")
            overlay(img, msk, cajas.get(iid, []), ov)
            w.writerow([iid, fila.get("n_boxes", ""), estratos[iid],
                        os.path.join("imagenes", f"{iid}.png"),
                        os.path.join("mascaras_sam2", f"{iid}.png"),
                        os.path.join("overlays", f"{iid}.png")])

    conteo = defaultdict(int)
    for i in elegidas:
        conteo[estratos[i["id"]]] += 1
    print(f"  Estratos: {dict(conteo)}")
    print(f"  Salida  : {a.out}")
    if faltantes:
        print(f"  [!!] sin imagen/máscara: {faltantes}")
        return 1
    print("  Listo. El validador corrige/aprueba en overlays/ y exporta"
          " PNG a mascaras_humano/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
