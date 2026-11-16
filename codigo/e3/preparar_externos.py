"""
PREPARACIÓN DE DATASETS EXTERNOS (Entregable 3, R3: >=3 datasets adicionales)

Lee un .zip SIN extraerlo completo (el de Montgomery pesa ~10 GB) y deja, para
cada dataset, imágenes y máscaras PNG de 512x512 en 1 canal (el mismo tamaño
que TBX11K) más un manifiesto CSV con SHA-256.

    # 1) ver qué hay dentro del zip (no escribe nada)
    python preparar_externos.py listar  datos/montgomery.zip

    # 2) zip "Lung Segmentation" (Montgomery MCU + Shenzhen CHN juntos)
    python preparar_externos.py pulmon datos/montgomery.zip \\
           --salida datos/externos

Salida:  <salida>/imgs/<id>.png, <salida>/masks/<id>.png ({0,255}),
         <salida>/manifest.csv (id, sha256_img, sha256_mask, forma_original)

OJO (declarar en el informe): Montgomery/Shenzhen/JSRT traen máscaras de
PULMÓN, no de lesión TB. Sirven como validación externa de la misma
modalidad (radiografía de tórax) y de la capacidad de segmentar anatomía
torácica; no miden segmentación de lesiones.
"""
import argparse
import collections
import csv
import hashlib
import io
import os
import re
import sys
import zipfile

TAM = 512


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def listar(zpath, maximo=12):
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
    ext = collections.Counter(os.path.splitext(i.filename)[1].lower()
                              for i in infos)
    carpetas = collections.Counter(os.path.dirname(i.filename) for i in infos)
    tam = sum(i.file_size for i in infos) / 1e9
    print(f"{len(infos)} archivos, {tam:.2f} GB descomprimidos")
    print("extensiones:", dict(ext.most_common()))
    print("carpetas con más archivos:")
    for c, n in carpetas.most_common(maximo):
        print(f"  {n:6d}  {c or '(raíz)'}")
    grandes = sorted(infos, key=lambda i: -i.file_size)[:5]
    print("5 más grandes:")
    for i in grandes:
        print(f"  {i.file_size / 1e6:9.1f} MB  {i.filename}")


def _a_512(arr, mascara):
    import numpy as np
    from PIL import Image
    im = Image.fromarray(arr)
    modo = Image.NEAREST if mascara else Image.BILINEAR
    return np.array(im.resize((TAM, TAM), modo))


def _leer(z, nombre):
    from PIL import Image
    with z.open(nombre) as f:
        return Image.open(io.BytesIO(f.read()))


def pulmon(zpath, salida_base):
    """Zip tipo 'Lung Segmentation' (Montgomery MCU + Shenzhen CHN juntos):
    CXR_png/<id>.png y masks/<id>[_mask].png con UNA máscara por imagen.
    Reparte por prefijo en <salida_base>/montgomery y <salida_base>/shenzhen.
    Ignora la copia duplicada bajo 'data/' y las imágenes sin máscara."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        sys.exit("Faltan paquetes. Ejecuta:  py -m pip install numpy pillow")
    destinos = {"MCU": "montgomery", "CHN": "shenzhen"}
    with zipfile.ZipFile(zpath) as z:
        nombres = [n for n in z.namelist() if n.lower().endswith(".png")]
        raices = [n for n in nombres if not n.startswith("data/")] or nombres
        imgs, masks = {}, {}
        for n in raices:
            bajo = n.replace("\\", "/").lower()
            stem = os.path.splitext(os.path.basename(n))[0]
            if "/cxr_png/" in bajo:
                imgs[stem] = n
            elif "/masks/" in bajo:
                masks[re.sub(r"_mask$", "", stem, flags=re.I)] = n
        ids = sorted(set(imgs) & set(masks))
        sin_mask = sorted(set(imgs) - set(masks))
        print(f"imágenes {len(imgs)}, máscaras {len(masks)}, "
              f"pares {len(ids)}, imágenes sin máscara {len(sin_mask)} "
              f"(se omiten)")
        if not ids:
            sys.exit("No hay pares imagen/máscara: ejecuta 'listar' y revisa "
                     "los nombres de CXR_png/ y masks/.")
        filas = {d: [] for d in destinos.values()}
        redim = 0
        for k, i in enumerate(ids, 1):
            pref = i[:3].upper()
            if pref not in destinos:
                print(f"  prefijo desconocido, se omite: {i}")
                continue
            d = destinos[pref]
            sal = os.path.join(salida_base, d)
            os.makedirs(os.path.join(sal, "imgs"), exist_ok=True)
            os.makedirs(os.path.join(sal, "masks"), exist_ok=True)
            ig = _leer(z, imgs[i])
            forma = ig.size
            ig = np.array(ig.convert("L"))
            mk = np.array(_leer(z, masks[i]).convert("L"))
            if mk.shape != ig.shape:
                redim += 1
                mk = np.array(Image.fromarray(mk).resize(
                    (ig.shape[1], ig.shape[0]), Image.NEAREST))
            mk = ((mk > 127) * 255).astype(np.uint8)
            ig8, mk8 = _a_512(ig, False), _a_512(mk, True)
            pi, pm = (os.path.join(sal, "imgs", f"{i}.png"),
                      os.path.join(sal, "masks", f"{i}.png"))
            Image.fromarray(ig8).save(pi)
            Image.fromarray(mk8).save(pm)
            filas[d].append({"id": i,
                             "sha256_img": _sha(open(pi, "rb").read()),
                             "sha256_mask": _sha(open(pm, "rb").read()),
                             "forma_original": f"{forma[0]}x{forma[1]}",
                             "frac_pulmon": round(float((mk8 > 0).mean()), 4)})
            if k % 50 == 0:
                print(f"  {k}/{len(ids)}", flush=True)
    for d, fs in filas.items():
        if not fs:
            continue
        with open(os.path.join(salida_base, d, "manifest.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fs[0]))
            w.writeheader()
            w.writerows(fs)
        print(f"{d}: {len(fs)} pares en {os.path.join(salida_base, d)}")
    if redim:
        print(f"aviso: {redim} máscaras tenían otro tamaño que su imagen "
              f"y se reescalaron (vecino más cercano)")


def a_npz(entrada, npz_dir, splits_dir, nombre, semilla=2026, frac=(0.70, 0.15)):
    """PNG 512 (imgs/masks) -> NPZ + CSV de splits en el formato de DatasetTB.

    Reparto fijo por semilla (70 % train, 15 % val de selección, 15 % test
    retenido). El test retenido se escribe como <splits>/heldout/val.csv para
    que DatasetTB('val', splits_dir=<...>/heldout) lo lea sin cambios.
    """
    import numpy as np
    from PIL import Image
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(entrada, "imgs"))
                 if f.lower().endswith(".png"))
    if not ids:
        sys.exit(f"sin imágenes en {entrada}/imgs")
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(ids))
    n_tr, n_va = int(frac[0] * len(ids)), int(frac[1] * len(ids))
    grupos = {"train": orden[:n_tr], "val": orden[n_tr:n_tr + n_va],
              "heldout": orden[n_tr + n_va:]}
    campos = ["id", "split", "origen", "gt_tipo", "tb_type", "label", "n_boxes",
              "mask_frac", "mask_area", "fname", "ruta"]
    for g, idx in grupos.items():
        carpeta = os.path.join(npz_dir, nombre, g)
        os.makedirs(carpeta, exist_ok=True)
        filas = []
        for k in idx:
            i = ids[k]
            im = np.array(Image.open(os.path.join(entrada, "imgs", i + ".png")).convert("L"))
            mk = (np.array(Image.open(os.path.join(entrada, "masks", i + ".png")).convert("L")) > 127)
            ruta = os.path.abspath(os.path.join(carpeta, i + ".npz"))
            np.savez_compressed(ruta, image=im.astype(np.float32), mask=mk.astype(np.float32))
            filas.append({"id": i, "split": g, "origen": "npz", "gt_tipo": "manual_pulmon",
                          "tb_type": "externo_" + nombre, "label": 1, "n_boxes": 0,
                          "mask_frac": round(float(mk.mean()), 6), "mask_area": int(mk.sum()),
                          "fname": i + ".png", "ruta": ruta})
        destino = splits_dir if g != "heldout" else os.path.join(splits_dir, "heldout")
        os.makedirs(destino, exist_ok=True)
        nombre_csv = "val.csv" if g == "heldout" else f"{g}.csv"
        with open(os.path.join(destino, nombre_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"  {nombre} {g:8s}: {len(filas)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("listar")
    a.add_argument("zip")
    b = sub.add_parser("pulmon", help="zip 'Lung Segmentation' (Montgomery+Shenzhen)")
    b.add_argument("zip")
    b.add_argument("--salida", default="datos/externos")
    c = sub.add_parser("npz", help="PNG 512 -> NPZ + splits en formato DatasetTB")
    c.add_argument("entrada", help="carpeta con imgs/ y masks/ (p. ej. datos/externos/montgomery)")
    c.add_argument("--nombre", required=True)
    c.add_argument("--npz-dir", default="datos/externos_npz")
    c.add_argument("--splits-dir", default=None,
                   help="por defecto datos/externos_splits/<nombre>")
    r = ap.parse_args()
    if r.cmd == "npz":
        a_npz(r.entrada, r.npz_dir, r.splits_dir or os.path.join("datos", "externos_splits", r.nombre), r.nombre)
        return
    if r.cmd == "listar":
        listar(r.zip)
    else:
        pulmon(r.zip, r.salida)


if __name__ == "__main__":
    main()
