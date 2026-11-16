"""
ESTADÍSTICA DEL ENTREGABLE 3 (R3): modelo original vs top 3 del benchmark.

Pruebas (sobre el test sellado, 80 positivas, Dice por imagen):
  T1  Wilcoxon de rangos con signo, pareado por imagen (media sobre semillas)
  T2  Bootstrap pareado por imagen de la diferencia de medias (IC 95 %)
  T3  Mann-Whitney sobre medias por semilla (>=10 vs >=10)
  Omnibus: Friedman sobre imagenes (original + 3 rivales)
Corrección: Holm sobre los 9 p-valores (3 pruebas x 3 rivales), alfa = 0.05.
Tamaño de efecto: delta de Cliff (|d| >= 0.147 = al menos pequeño) y
diferencia de medias frente al margen de equivalencia declarado (0.01).

Salida: <resultados>/analisis_e3/comparaciones_e3.csv  y  resumen_e3.md
Uso:   python estadistica_e3.py --resultados resultados_e3
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

MARGEN = 0.01
UMBRAL_CLIFF = 0.147


def cliff_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mayor = (x[:, None] > y[None, :]).sum()
    menor = (x[:, None] < y[None, :]).sum()
    return float((mayor - menor) / (len(x) * len(y)))


def holm(p):
    p = np.asarray(p, float)
    orden = np.argsort(p)
    aj = np.empty_like(p)
    previo = 0.0
    m = len(p)
    for rango, i in enumerate(orden):
        previo = max(previo, (m - rango) * p[i])
        aj[i] = min(1.0, previo)
    return aj


def cargar(res_dir, modelo, semillas, archivo="por_imagen_test.csv", split="test_pos"):
    """-> (ids, matriz [n_semillas_ok, n_imagenes] de Dice en test_pos)."""
    filas, ids_ref = [], None
    for s in semillas:
        p = os.path.join(res_dir, modelo, f"seed_{s}", archivo)
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            r = [x for x in csv.DictReader(f) if not split or x.get("split") == split]
        r.sort(key=lambda x: x["id"])
        ids = [x["id"] for x in r]
        if ids_ref is None:
            ids_ref = ids
        if ids != ids_ref:
            raise ValueError(f"{modelo} semilla {s}: conjunto de imágenes distinto")
        filas.append([float(x["dice"]) for x in r])
    if not filas:
        raise FileNotFoundError(f"sin por_imagen_test.csv para {modelo}")
    return ids_ref, np.array(filas)


def bootstrap_pareado(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    medias = d[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    p = 2 * min((medias <= 0).mean(), (medias >= 0).mean())
    return float(lo), float(hi), float(min(1.0, max(p, 1.0 / n)))


def comparar(res_dir, original, rivales, semillas, archivo="por_imagen_test.csv", split="test_pos"):
    ids, mo = cargar(res_dir, original, semillas, archivo, split)
    datos = {original: mo}
    for r in rivales:
        i2, m2 = cargar(res_dir, r, semillas, archivo, split)
        if i2 != ids:
            raise ValueError(f"{r}: imágenes distintas a las de {original}")
        datos[r] = m2
    img = {m: v.mean(axis=0) for m, v in datos.items()}          # por imagen
    sem = {m: v.mean(axis=1) for m, v in datos.items()}          # por semilla
    filas = []
    for r in rivales:
        d = img[original] - img[r]
        w = stats.wilcoxon(img[original], img[r], zero_method="wilcox")
        lo, hi, pb = bootstrap_pareado(d)
        if len(sem[original]) >= 2 and len(sem[r]) >= 2:
            mw = stats.mannwhitneyu(sem[original], sem[r], alternative="two-sided")
            p_mw, dsem = float(mw.pvalue), cliff_delta(sem[original], sem[r])
        else:
            p_mw, dsem = float("nan"), float("nan")
        base = {"rival": r, "n_img": len(ids),
                "n_sem_orig": len(sem[original]), "n_sem_rival": len(sem[r]),
                "dice_orig": float(img[original].mean()),
                "dice_rival": float(img[r].mean()),
                "dif_media": float(d.mean())}
        filas.append({**base, "prueba": "T1 Wilcoxon (imagen)", "p": float(w.pvalue),
                      "cliff": cliff_delta(img[original], img[r]), "ic95": ""})
        filas.append({**base, "prueba": "T2 Bootstrap pareado", "p": pb,
                      "cliff": cliff_delta(img[original], img[r]),
                      "ic95": f"[{lo:.4f}, {hi:.4f}]"})
        filas.append({**base, "prueba": "T3 Mann-Whitney (semillas)", "p": p_mw,
                      "cliff": dsem, "ic95": ""})
    validos = [i for i, f in enumerate(filas) if not np.isnan(f["p"])]
    aj = holm([filas[i]["p"] for i in validos])
    for i, pa in zip(validos, aj):
        filas[i]["p_holm"] = float(pa)
    for f in filas:
        f.setdefault("p_holm", float("nan"))
        f["significativo_holm"] = bool(f["p_holm"] < 0.05)
        f["efecto_relevante"] = bool(abs(f["cliff"]) >= UMBRAL_CLIFF) \
            if not np.isnan(f["cliff"]) else False
        f["supera_margen"] = bool(f["dif_media"] > MARGEN)
        f["gana_original"] = bool(f["significativo_holm"] and f["efecto_relevante"]
                                  and f["dif_media"] > 0)
    fr = stats.friedmanchisquare(*[img[m] for m in datos]) if len(datos) >= 3 else None
    return filas, fr, datos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--original", default="lacca_unet")
    ap.add_argument("--rivales", nargs="*", default=None,
                    help="por defecto: top de <resultados>/top3.json")
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--archivo", default="por_imagen_test.csv",
                    help="CSV por imagen (externos: por_imagen_externo.csv)")
    ap.add_argument("--split", default="test_pos",
                    help="valor de la columna 'split' a usar ('' = sin filtro)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modelos_tb import SEMILLAS, SEMILLAS_EXTRA
    semillas = a.semillas or (SEMILLAS + SEMILLAS_EXTRA)
    rivales = a.rivales or json.load(open(os.path.join(
        a.resultados, "top3.json"), encoding="utf-8"))["top"]
    filas, fr, datos = comparar(a.resultados, a.original, rivales, semillas, a.archivo, a.split)
    out = os.path.join(a.resultados, "analisis_e3")
    os.makedirs(out, exist_ok=True)
    campos = ["rival", "prueba", "n_img", "n_sem_orig", "n_sem_rival", "dice_orig",
              "dice_rival", "dif_media", "p", "p_holm", "cliff", "ic95",
              "significativo_holm", "efecto_relevante", "supera_margen",
              "gana_original"]
    with open(os.path.join(out, "comparaciones_e3.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for x in filas:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in x.items() if k in campos})
    L = ["# Comparaciones del modelo original (E3)", "",
         f"Original: `{a.original}`; rivales: {', '.join(rivales)}. "
         f"Semillas por modelo: "
         f"{ {m: v.shape[0] for m, v in datos.items()} }.", ""]
    if fr is not None:
        L.append(f"Friedman (imágenes): chi2={fr.statistic:.3f}, p={fr.pvalue:.4g}\n")
    L += ["| rival | prueba | Δ Dice | p | p Holm | Cliff δ | IC 95 % | gana |",
          "|---|---|---|---|---|---|---|---|"]
    for x in filas:
        L.append(f"| {x['rival']} | {x['prueba']} | {x['dif_media']:+.4f} | "
                 f"{x['p']:.4g} | {x['p_holm']:.4g} | {x['cliff']:+.3f} | "
                 f"{x['ic95']} | {'sí' if x['gana_original'] else 'no'} |")
    L += ["", "Holm sobre los 9 p-valores. 'gana' = significativo tras Holm, "
          f"|δ| >= {UMBRAL_CLIFF} y diferencia a favor del original."]
    open(os.path.join(out, "resumen_e3.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
