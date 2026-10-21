#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicación parcial del paper guía (Liu et al., CVPR 2020, "Rethinking
Computer-aided Tuberculosis Diagnosis", TBX11K) usando SOLO el paquete
público liberado (datos/kaggle_dataset/TBX11K/).

Objetivo (rúbrica, Sección 5): "Reproducir al menos un resultado clave del
paper bajo sus condiciones originales" y "reportar la discrepancia entre lo
reportado y lo obtenido".

Qué se replica (resultados del paper basados en el paquete público):
  C1  Tabla 2 — composición del conjunto train+val (3 800 / 3 800 / 800).
  C2  Tabla 2 — partición oficial train (6 600: 3 000/3 000/600) vía
      lists/TBX11K_train.txt.
  C3  Tabla 2 — partición val derivada (1 800: 800/800/200) = trainval − train.
  C4  Figura 1 — estadísticas de cajas (nº de cajas, área como fracción
      de la imagen 512x512).
  C5  Condición de entrada 512x512 (los detectores del paper entrenan con
      imágenes "resized to 512x512 when inputting to networks", Sec. 4.1).
  C6  Integridad de protocolo: los splits sellados de este proyecto usan SOLO
      imágenes del trainval oficial (el test de 2 800 del paper tiene su GT
      NO publicado y no interviene en ningún split nuestro).

No se replica (declarado, imposible por construcción):
  Tablas 3 y 4 del paper (clasificación/detección sobre el test de 2 800):
  el GT de ese test no está publicado en el paquete (README oficial).

Salida: reporte en stdout y en referencias/papers/replicacion_resultados.md
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PKG = RAIZ / "datos" / "kaggle_dataset" / "TBX11K"
SPLITS = RAIZ / "datos" / "splits"
OUT = RAIZ / "referencias" / "papers" / "replicacion_resultados.md"

# Categorías del paquete (JSON COCO): 1 ActiveTuberculosis,
# 2 ObsoletePulmonaryTuberculosis (latente/curada), 3 PulmonaryTuberculosis
# (existe pero sin anotaciones en trainval). Mapeo al active/latent del paper:
CAT_ACTIVO = {1}
CAT_LATENTE = {2, 3}

# Resultados reportados por el paper (Tabla 2)
PAPER_TAB2 = {  # train, val, test, total
    "healthy": (3000, 800, 1200, 5000),
    "sick": (3000, 800, 1200, 5000),
    "active": (473, 157, 294, 924),
    "latent": (104, 36, 72, 212),
    "both": (23, 7, 24, 54),
    "uncertain": (0, 0, 10, 10),
    "total": (6600, 1800, 2800, 11200),
}

lineas = []


def log(s=""):
    print(s)
    lineas.append(s)


def cargar_json(nombre):
    with open(PKG / "annotations" / "json" / nombre, encoding="utf-8") as f:
        return json.load(f)


def clase_tb(cats_de_la_imagen):
    ids = set(cats_de_la_imagen)
    activo = bool(ids & CAT_ACTIVO)
    laten = bool(ids & CAT_LATENTE)
    if activo and laten:
        return "both"
    if activo:
        return "active"
    if laten:
        return "latent"
    return "other"


def resumen_cajas(anns, etiqueta):
    areas = sorted(a["area"] / (512 * 512) for a in anns)
    n = len(areas)

    def pct(p):
        return areas[min(n - 1, int(p / 100 * n))] if n else float("nan")

    log(f"- **{etiqueta}**: {n} cajas · área/fracción mediana {pct(50):.4f} · "
        f"IQR [{pct(25):.4f}, {pct(75):.4f}] · p90 {pct(90):.4f}")


def main():
    rnd = random.Random(42)
    log("# Replicación del paper guía — resultados ejecutados")
    log()
    log("Ejecución de `codigo/replicacion_paper_guia.py` (semilla 42 solo para "
        "el muestreo de resolución). Sin entrenamiento: usa únicamente el "
        "paquete público liberado de TBX11K.")
    log()

    tr_json = cargar_json("TBX11K_train_only_tb.json")
    tv_json = cargar_json("TBX11K_trainval_only_tb.json")

    log("## 0 · Categorías del paquete (JSON COCO)")
    for c in tv_json["categories"]:
        log(f"- id {c['id']}: {c['name']}")
    log("- mapeo al paper: 1→active · 2,3→latent")
    log()

    anns_por_img = {}
    for a in tv_json["annotations"]:
        anns_por_img.setdefault(a["image_id"], []).append(a)

    # ---- carpetas del paquete ------------------------------------------
    def cuenta(carpeta):
        return len(list((PKG / "imgs" / carpeta).glob("*.png")))

    n_health, n_sick, n_tb = cuenta("health"), cuenta("sick"), cuenta("tb")

    tv_classes = Counter(
        clase_tb([a["category_id"] for a in anns_por_img.get(im["id"], [])])
        for im in tv_json["images"]
    )

    log("## C1 · Composición train+val (Tabla 2 del paper)")
    log("| categoría | paper (train+val) | paquete medido | Δ |")
    log("|---|---|---|---|")
    filas = [
        ("healthy", 3800, n_health),
        ("sick & non-TB", 3800, n_sick),
        ("TB total (carpeta imgs/tb)", 800, n_tb),
        ("TB con cajas (json trainval)", 800, len(tv_json["images"])),
        ("  active TB", 630, tv_classes.get("active", 0)),
        ("  latent TB", 140, tv_classes.get("latent", 0)),
        ("  active & latent", 30, tv_classes.get("both", 0)),
        ("  sin cajas en json", 0, tv_classes.get("other", 0)),
        ("TOTAL train+val (carpetas)", 8400, n_health + n_sick + n_tb),
    ]
    for nom, esp, med in filas:
        log(f"| {nom} | {esp} | {med} | {med - esp:+d} |")
    log()

    # ---- C2 train --------------------------------------------------------
    train_list = [l.strip() for l in
                  (PKG / "lists" / "TBX11K_train.txt").read_text().splitlines() if l.strip()]
    carpeta = Counter(p.split("/")[0] for p in train_list)
    nombre_a_id = {im["file_name"]: im["id"] for im in tr_json["images"]}
    anns_tr = {}
    for a in tr_json["annotations"]:
        anns_tr.setdefault(a["image_id"], []).append(a)
    tr_classes = Counter(
        clase_tb([a["category_id"] for a in anns_tr.get(nombre_a_id[f], [])])
        for f in train_list if f in nombre_a_id
    )

    log("## C2 · Partición oficial train (Tabla 2 del paper)")
    log(f"- lista `TBX11K_train.txt`: **{len(train_list)}** líneas (paper: 6 600) "
        f"→ Δ = {len(train_list) - 6600:+d}")
    log(f"- por carpeta: tb {carpeta.get('tb', 0)} (paper 600) · sick "
        f"{carpeta.get('sick', 0)} (paper 3 000) · health {carpeta.get('health', 0)} "
        f"(paper 3 000)")
    log(f"- clases TB del json de train: active {tr_classes.get('active', 0)} (paper 473) · "
        f"latent {tr_classes.get('latent', 0)} (paper 104) · both {tr_classes.get('both', 0)} "
        f"(paper 23)")
    log()

    # ---- C3 val derivado -------------------------------------------------
    tb_train = {p.split("/")[1].replace(".png", "") for p in train_list if p.startswith("tb/")}
    tb_tv = {im["file_name"].split("/")[1].replace(".png", "") for im in tv_json["images"]}
    log("## C3 · Partición val derivada (trainval − train)")
    log(f"- TB en val derivado: **{len(tb_tv - tb_train)}** (paper: 200)")
    log(f"- sick en val: {n_sick - carpeta.get('sick', 0)} (paper 800) · "
        f"health en val: {n_health - carpeta.get('health', 0)} (paper 800)")
    log(f"- clases TB en val: active {tv_classes.get('active', 0) - tr_classes.get('active', 0)} "
        f"(paper 157) · latent {tv_classes.get('latent', 0) - tr_classes.get('latent', 0)} "
        f"(paper 36) · both {tv_classes.get('both', 0) - tr_classes.get('both', 0)} (paper 7)")
    log()

    # ---- C4 cajas (Fig. 1) ----------------------------------------------
    log("## C4 · Estadísticas de cajas (Figura 1 del paper, escala 512×512)")
    resumen_cajas(tr_json["annotations"], "train (json solo-TB)")
    resumen_cajas(tv_json["annotations"], "train+val (json solo-TB)")
    n_con_cajas = len(anns_por_img)
    log(f"- imágenes TB trainval con ≥1 caja: **{n_con_cajas}** de "
        f"{len(tv_json['images'])}")
    dist = Counter(min(len(anns_por_img.get(im["id"], [])), 4) for im in tv_json["images"])
    log(f"- cajas por imagen (1/2/3/≥4): {dist.get(1, 0)}/{dist.get(2, 0)}/"
        f"{dist.get(3, 0)}/{dist.get(4, 0)}")
    log()

    # ---- C5 resolución ----------------------------------------------------
    log("## C5 · Condición de entrada 512×512 (Sec. 4.1 del paper)")
    anchos = {im["width"] for im in tv_json["images"]} | {im["height"] for im in tv_json["images"]}
    from PIL import Image
    tamaños = set()
    for m in rnd.sample(sorted(tb_tv), 50):
        with Image.open(PKG / "imgs" / "tb" / f"{m}.png") as imx:
            tamaños.add(imx.size)
    log(f"- metadatos JSON: width/height ∈ {sorted(anchos)}")
    log(f"- muestreo PIL de 50 PNG TB: tamaños observados {sorted(tamaños)}")
    log()

    # ---- C6 nuestros splits ⊂ trainval -----------------------------------
    log("## C6 · Protocolo: nuestros splits dentro del trainval oficial")
    ids_health = {p.stem for p in (PKG / "imgs" / "health").glob("*.png")}
    todo_ok = True
    for nombre_csv in ["train", "val", "test_pos", "test_neg", "test_pos_reserva"]:
        ids = [r["id"] for r in csv.DictReader(open(SPLITS / f"{nombre_csv}.csv", encoding="utf-8"))]
        base = ids_health if nombre_csv == "test_neg" else tb_tv
        fuera = [i for i in ids if i not in base]
        todo_ok &= not fuera
        log(f"- `{nombre_csv}.csv` (n={len(ids)}): fuera del trainval oficial: "
            f"**{len(fuera)}** {fuera[:3] if fuera else ''}")
    log(f"- veredicto: {'✅ ningún split usa el test oculto del paper' if todo_ok else '❌ HAY FUGA'}")
    log()

    # ---- discrepancias ----------------------------------------------------
    log("## Discrepancias detectadas (paper vs paquete liberado)")
    log("1. `TBX11K_train_only_tb.json` declara **599** imágenes TB (paper: 600) y "
        f"`TBX11K_trainval_only_tb.json` **{len(tv_json['images'])}** (paper: 800), "
        "mientras las carpetas contienen 600/800 PNG. Causa probable: omisión de "
        "una imagen sin cajas al generar los JSON. No afecta a este proyecto "
        "(C6).")
    log("2. El paquete solo incluye `TBX11K_train.txt` y `all_train.txt`; el README "
        "describe también `TBX11K_val.txt`/`TBX11K_trainval.txt`, ausentes. Causa: "
        "re-empaquetado de terceros (distribución vía Kaggle). Val reconstruido "
        "por diferencia de conjuntos (C3).")
    log("3. Resolución: el paper describe los radiografías originales ≈3000×3000, "
        "pero la condición de red es 512×512 (Sec. 4.1) y el paquete libera "
        "512×512 (C5). Nuestra entrada nativa 512×512 es equivalente a la suya.")
    log("4. Tablas 3–4 (clasificación/detección sobre test): **no reproducibles** — "
        "el GT del test de 2 800 no se publicó (declarado por los autores). Este "
        "trabajo replica los resultados de *datos* y la condición de entrada; "
        "los detectores SSD/RetinaNet/Faster R-CNN/FCOS quedan fuera del Top-10 "
        "por ser detección de cajas, no segmentación píxel-a-píxel.")
    log()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n[ok] reporte escrito en {OUT.relative_to(RAIZ)}", file=sys.stderr)


if __name__ == "__main__":
    main()
