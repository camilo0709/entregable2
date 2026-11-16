"""Prepara JSRT (247 radiografías, imágenes .IMG + máscaras SCR de pulmón) al mismo
formato que Montgomery/Shenzhen: <salida>/imgs/<id>.png, masks/<id>.png ({0,255}), manifest.csv.

Descarga (manual): imágenes  http://db.jsrt.or.jp/eng.php  (carpeta con JPCLN###.IMG / JPCNN###.IMG)
                   máscaras  https://www.isi.uu.nl/Research/Databases/SCR/  (scratch/masks/{left lung,right lung}/*.gif)
Uso:  py codigo\\preparar_jsrt.py --imagenes RUTA\\IMG --mascaras RUTA\\masks --salida datos\\externos\\jsrt
.IMG = 2048x2048, uint16 big-endian (12 bits). Máscara = unión pulmón izquierdo + derecho.
"""
import argparse, csv, hashlib, os, sys
import numpy as np
from PIL import Image

def leer_img(p):
    a = np.fromfile(p, dtype=">u2")
    if a.size != 2048 * 2048:
        raise ValueError(f"{p}: {a.size} valores, se esperaban {2048*2048}")
    a = a.reshape(2048, 2048).astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))

def leer_mask(dir_m, i):
    m = None
    for lado in ("left lung", "right lung"):
        for ext in (".gif", ".png"):
            p = os.path.join(dir_m, lado, i + ext)
            if os.path.exists(p):
                x = np.array(Image.open(p).convert("L")) > 127
                m = x if m is None else (m | x); break
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenes", required=True); ap.add_argument("--mascaras", required=True)
    ap.add_argument("--salida", default="datos/externos/jsrt")
    a = ap.parse_args()
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(a.imagenes) if f.upper().endswith(".IMG"))
    if not ids: sys.exit("sin .IMG en --imagenes")
    os.makedirs(os.path.join(a.salida, "imgs"), exist_ok=True); os.makedirs(os.path.join(a.salida, "masks"), exist_ok=True)
    filas, omit = [], []
    for i in ids:
        mk = leer_mask(a.mascaras, i)
        if mk is None: omit.append(i); continue
        ig = leer_img(os.path.join(a.imagenes, i + ".IMG"))
        ig = ig.resize((512, 512), Image.BILINEAR)
        mk = Image.fromarray((mk * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
        pi, pm = os.path.join(a.salida, "imgs", i + ".png"), os.path.join(a.salida, "masks", i + ".png")
        ig.save(pi); mk.save(pm)
        h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
        filas.append([i, h(pi), h(pm), "2048x2048", round(float((np.array(mk) > 127).mean()), 4)])
    with open(os.path.join(a.salida, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "sha256_img", "sha256_mask", "forma_original", "frac_pulmon"]); w.writerows(filas)
    print(f"JSRT: {len(filas)} listas, {len(omit)} sin máscara {omit[:5]}")
    if len(filas) != 247: print("AVISO: se esperaban 247")

if __name__ == "__main__": main()
