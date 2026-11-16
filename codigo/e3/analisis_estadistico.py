"""
================================================================================
ANÁLISIS ESTADÍSTICO DEL BENCHMARK - ENTREGABLE 2
================================================================================

PROPÓSITO:
    Consumir `resultados/resumen_test.csv` + los `por_imagen_test.csv` de
    cada run y producir EXACTAMENTE lo preregistrado (docs/PREREGISTRO_
    ENTREGABLE2.md):

        Tabla A  descriptiva por modelo (Dice ± DE entre semillas, IoU,
                 HD95 con denominador, ASSD, parámetros, s/época, AUC)
        Tabla B  ranking global (Friedman + Iman-Davenport)
        Tabla C  comparaciones vs el mejor (Δ Dice, IC 95% BCa, Wilcoxon
                 con Holm, r = Z/√N, decisión mejor/equivalente/peor)
        H1.c     DeLong (AUC de detección) y McNemar exacto

    Incluye:
      - Baseline triviales `empty_mask` y `bbox_mask` (calculados aquí,
        sin checkpoint; `bbox_mask` usa las cajas COCO originales de TBX11K).
      - Análisis de sensibilidad: Friedman repetido por semilla.
      - Aviso automático de modelos con predicciones duplicadas.

DEFINICIONES FIJADAS:
    - Unidad de análisis: imagen (n=80 test_pos).
    - Cada modelo se promedia sobre sus 5 semillas ANTES del análisis por
      pares (preregistro §4).
    - δ = 0.01 en Dice para equivalencia.
    - BCa con >= 2000 remuestreos (por defecto 10000), semilla 42.
    - Holm-Bonferroni en las 9 comparaciones.
    - DeLong: score = masa de probabilidad (suma sigmoid / 1000).
    - McNemar: acierto binario por imagen (¿algún píxel > 0.5?) frente a
      presencia real, en las 120 imágenes (80 pos + 40 neg).

USO:
    python codigo/analisis_estadistico.py
    python codigo/analisis_estadistico.py --n-bootstrap 20000
    python codigo/analisis_estadistico.py --excluir attention_unet

SALIDAS (en resultados/analisis/):
    Tabla_A.csv, Tabla_B.csv, Tabla_C.csv, Tabla_D_deteccion.csv,
    sensibilidad_friedman.csv, matrices_dice.csv, analisis_estadistico.json,
    INFORME_ESTADISTICO.md
================================================================================
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modelos_tb import MODELOS, ORDEN_BENCHMARK, ORDEN_BASELINES, SEMILLAS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR_DEF = os.path.join(ROOT, "resultados")
SPLITS_DEF = os.path.join(ROOT, "datos", "splits")
COCO_DIR = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K",
                        "annotations", "json")
DELTA = 0.01
ALPHA = 0.05
N_BOOT = 10000
SEED = 42

# Aclaraciones que deben viajar con la tabla (enmiendas E1-E4 y sustitutos
# declarados en el preregistro). No alteran ningún número: solo etiquetan.
NOTAS = {
    "attention_unet": "re-entrenado tras corregir el gate (E1)",
    "nnunet": "pipeline nnU-Net 2d; 5 folds <-> 5 semillas (E3)",
    "transunet": "sustituto declarado: smp.Unet resnet50 + scse",
    "swin_unet": "sustituto declarado: smp.Unet mit_b2",
    "mednext": "sustituto declarado: MONAI SegResNet",
    "unet_fuerte": "duplicado de unet (E4); excluido del Friedman",
    "empty_mask": "baseline sin red (cota inferior)",
    "bbox_mask": "baseline sin red (cajas TBX11K)",
}


# ============================================================================
# CARGA
# ============================================================================

def leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "si", "sí")


def _float(v):
    try:
        if v is None or v == "" or str(v).lower() == "none":
            return None
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def ids_split(splits_dir, split):
    return [r["id"] for r in leer_csv(os.path.join(splits_dir, f"{split}.csv"))]


def cargar_predicciones(res_dir, modelos, semillas, splits_dir):
    """
    Devuelve:
        datos[modelo][semilla][id] = dict con dice, iou, hd95, assd,
        pred_any, prob_masa, split
    y los ids de pos/neg.
    """
    ids_pos = ids_split(splits_dir, "test_pos")
    ids_neg = ids_split(splits_dir, "test_neg")
    set_pos, set_neg = set(ids_pos), set(ids_neg)

    datos, faltantes = {}, {}
    for m in modelos:
        datos[m] = {}
        for s in semillas:
            rcsv = os.path.join(res_dir, m, f"seed_{s}", "por_imagen_test.csv")
            if not os.path.exists(rcsv):
                faltantes.setdefault(m, []).append(s)
                continue
            filas = {}
            for r in leer_csv(rcsv):
                iid = r["id"]
                split = r.get("split") or ("test_pos" if iid in set_pos
                                           else "test_neg" if iid in set_neg
                                           else "")
                if not split:
                    continue
                pred_any = r.get("pred_any")
                if pred_any in (None, ""):
                    pred_any = not _bool(r.get("pred_vacia", "False"))
                filas[iid] = {
                    "split": split,
                    "dice": _float(r.get("dice")),
                    "iou": _float(r.get("iou")),
                    "hd95": _float(r.get("hd95")),
                    "assd": _float(r.get("assd")),
                    "pred_any": _bool(pred_any),
                    "prob_masa": _float(r.get("prob_masa")),
                }
            datos[m][s] = filas
    return datos, ids_pos, ids_neg, faltantes


# ============================================================================
# BASELINES SIN RED
# ============================================================================

def _cajas_tbx11k():
    """id (stem) -> lista de cajas [x,y,w,h] de los COCO de TBX11K."""
    cajas = {}
    if not os.path.isdir(COCO_DIR):
        return cajas
    for nombre in ("TBX11K_trainval_only_tb.json", "TBX11K_val.json",
                   "TBX11K_train_only_tb.json", "all_test.json"):
        ruta = os.path.join(COCO_DIR, nombre)
        if not os.path.exists(ruta):
            continue
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        por_img = {im["id"]: im["file_name"] for im in d.get("images", [])}
        for an in d.get("annotations", []):
            fn = por_img.get(an.get("image_id"))
            if not fn:
                continue
            stem = os.path.splitext(os.path.basename(fn.replace("\\", "/")))[0]
            bb = an.get("bbox")
            if not bb or len(bb) != 4:
                continue
            cajas.setdefault(stem, [])
            c = [float(x) for x in bb]
            if c not in cajas[stem]:
                cajas[stem].append(c)
    return cajas


def _metrica_baseline(pred, gt):
    from metricas_tb import metricas_imagen
    return metricas_imagen(pred, gt)


def calcular_baselines(splits_dir, modelos_incluir=("empty_mask", "bbox_mask")):
    """Calcula las filas por imagen de los baselines sin red."""
    import numpy as np
    from PIL import Image

    neg = np.zeros((512, 512), np.uint8)
    salida = {m: {} for m in modelos_incluir}
    cajas = _cajas_tbx11k() if "bbox_mask" in modelos_incluir else {}
    sin_caja = []

    for split in ("test_pos", "test_neg"):
        for fila in leer_csv(os.path.join(splits_dir, f"{split}.csv")):
            iid = fila["id"]
            gt = neg if split == "test_neg" else _leer_gt(iid)
            if split == "test_pos" and gt is None:
                continue

            if "empty_mask" in salida:
                m = _metrica_baseline(neg, gt)
                m.update({"split": split, "pred_any": False,
                          "prob_masa": None})
                salida["empty_mask"][iid] = m

            if "bbox_mask" in salida:
                pred = neg.copy()
                if split == "test_pos":
                    if not cajas.get(iid):
                        sin_caja.append(iid)
                    for (x, y, w, h) in cajas.get(iid, []):
                        x0 = max(0, int(round(x)))
                        y0 = max(0, int(round(y)))
                        x1 = min(512, int(round(x + w)))
                        y1 = min(512, int(round(y + h)))
                        if x1 > x0 and y1 > y0:
                            pred[y0:y1, x0:x1] = 1
                m = _metrica_baseline(pred, gt)
                m.update({"split": split, "pred_any": bool(pred.any()),
                          "prob_masa": None})
                salida["bbox_mask"][iid] = m
    if sin_caja:
        print(f"   ⚠ bbox_mask: {len(sin_caja)} test_pos sin caja COCO "
              f"({', '.join(sin_caja[:5])}…) → predicción vacía")
    return salida


def _leer_gt(iid):
    from PIL import Image
    ruta = os.path.join(ROOT, "datos", "ground_truth_test", f"{iid}.png")
    if not os.path.exists(ruta):
        return None
    return (np.asarray(Image.open(ruta).convert("L")) > 127).astype(np.uint8)


# ============================================================================
# ESTADÍSTICA
# ============================================================================

def friedman_iman(matriz):
    """matriz (k modelos × n imágenes), sin NaN."""
    k, n = matriz.shape
    chi2, p_f = stats.friedmanchisquare(*[matriz[i] for i in range(k)])
    den = n * (k - 1) - chi2
    if den <= 0:
        return {"chi2": float(chi2), "p_friedman": float(p_f),
                "F_iman": None, "p_iman": None, "k": k, "n": n}
    F = (n - 1) * chi2 / den
    p_i = float(stats.f.sf(F, k - 1, (k - 1) * (n - 1)))
    return {"chi2": float(chi2), "p_friedman": float(p_f),
            "F_iman": float(F), "p_iman": p_i, "k": int(k), "n": int(n)}


def rangos_medios(matriz):
    """Rango medio de cada modelo (1 = mejor Dice)."""
    return stats.rankdata(-matriz, axis=0).mean(axis=1)


def holm(pvals):
    """Holm-Bonferroni. Devuelve array de p ajustadas en el orden original."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    adj = np.empty(m)
    acum = 0.0
    for i, idx in enumerate(orden):
        acum = max(acum, min(1.0, p[idx] * (m - i)))
        adj[idx] = acum
    return adj


def bca_ic(d, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """
    IC 95% BCa de la media de d (diferencias pareadas a nivel de imagen).

    Implementación estándar (Efron & Tibshirani): z0 por proporción de
    bootstrap por debajo de theta_hat, aceleración por jackknife.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return None, None, None
    theta = float(np.mean(d))
    if np.allclose(d, d[0]):
        return theta, theta, theta

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)

    prop = np.clip(np.mean(boot < theta), 1.0 / n_boot, 1 - 1.0 / n_boot)
    z0 = stats.norm.ppf(prop)

    # jackknife
    total = d.sum()
    theta_j = (total - d) / (n - 1)
    dif = theta_j.mean() - theta_j
    den = 6.0 * (np.sum(dif ** 2) ** 1.5)
    a = float(np.sum(dif ** 3) / den) if den > 0 else 0.0

    def _alpha(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    a_lo = float(np.clip(_alpha(stats.norm.ppf(alpha / 2)), 1e-4, 1 - 1e-4))
    a_hi = float(np.clip(_alpha(stats.norm.ppf(1 - alpha / 2)), 1e-4, 1 - 1e-4))
    lo = float(np.quantile(boot, a_lo))
    hi = float(np.quantile(boot, a_hi))
    return theta, lo, hi


def wilcoxon_vs(d, alpha=ALPHA):
    """Wilcoxon bilateral de d contra 0 + tamaño de efecto r = Z/√N."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return {"p": None, "z": None, "r": None}
    if np.allclose(d, 0):
        return {"p": 1.0, "z": 0.0, "r": 0.0}
    try:
        res = stats.wilcoxon(d, zero_method="wilcox", correction=False,
                             alternative="two-sided")
        p = float(res.pvalue)
    except ValueError:
        return {"p": 1.0, "z": 0.0, "r": 0.0}
    z = float(stats.norm.isf(max(p, 1e-300) / 2))
    if np.median(d) < 0:
        z = -z
    return {"p": p, "z": z, "r": float(z / np.sqrt(n))}


def auc_de_long(grupo_pos, grupo_neg):
    """AUC + componentes V10/V01 (DeLong) para un score continuo."""
    pos = np.asarray([x for x in grupo_pos if x is not None], dtype=float)
    neg = np.asarray([x for x in grupo_neg if x is not None], dtype=float)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None
    todos = np.concatenate([pos, neg])
    orden = stats.rankdata(todos, method="average")
    r_pos = orden[:n1]
    auc = (r_pos.sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    # V10: proporción de negativos con score menor + 0.5 empates
    v10 = np.array([np.mean((neg < x) + 0.5 * (neg == x)) for x in pos])
    v01 = np.array([np.mean((pos > y) + 0.5 * (pos == y)) for y in neg])
    return {"auc": float(auc), "v10": v10, "v01": v01,
            "n_pos": n1, "n_neg": n0}


def delong_pareado(a, b):
    """Test de DeLong para dos AUC pareadas (mismas imágenes)."""
    if a is None or b is None:
        return None
    n1, n0 = a["n_pos"], a["n_neg"]
    if n1 < 2 or n0 < 2:
        return None
    s10 = np.cov(np.vstack([a["v10"], b["v10"]]), ddof=1)[0, 1]
    s01 = np.cov(np.vstack([a["v01"], b["v01"]]), ddof=1)[0, 1]
    var = s10 / n1 + s01 / n0
    if var <= 0:
        return {"z": 0.0, "p": 1.0}
    z = (a["auc"] - b["auc"]) / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"z": float(z), "p": p}


def mcnemar_exacto(correcto_a, correcto_b):
    """McNemar exacto sobre aciertos binarios pareados."""
    a = np.asarray(correcto_a, dtype=bool)
    b = np.asarray(correcto_b, dtype=bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return {"p": 1.0, "n10": n10, "n01": n01}
    p = float(stats.binomtest(min(n10, n01), n10 + n01, 0.5,
                              alternative="two-sided").pvalue)
    return {"p": min(1.0, p), "n10": n10, "n01": n01}


# ============================================================================
# AGREGADOS
# ============================================================================

def media_por_semilla(datos_modelo, ids, semillas, clave):
    """media por semilla -> (media entre semillas, DE, vector por semilla)."""
    vals = []
    for s in semillas:
        filas = datos_modelo.get(s, {})
        v = [filas[i][clave] for i in ids
             if i in filas and filas[i].get(clave) is not None]
        vals.append(float(np.mean(v)) if v else np.nan)
    vals = np.asarray(vals, float)
    ok = ~np.isnan(vals)
    if not ok.any():
        return None, None, vals
    return float(vals[ok].mean()), float(vals[ok].std(ddof=0)), vals


def contar_definido(datos_modelo, ids, semillas, clave):
    """n medio de imágenes con la métrica definida, sobre las semillas con
    datos (los baselines solo tienen una)."""
    por_semilla = []
    for s in semillas:
        filas = datos_modelo.get(s, {})
        if not filas:
            continue
        por_semilla.append(sum(1 for i in ids
                               if i in filas and filas[i].get(clave) is not None))
    return int(round(float(np.mean(por_semilla)))) if por_semilla else 0


def vector_dice(datos_modelo, ids, semillas):
    """Dice medio por imagen (promedio entre semillas) -> vector n."""
    out = []
    for i in ids:
        vals = [datos_modelo[s][i]["dice"] for s in semillas
                if i in datos_modelo.get(s, {})
                and datos_modelo[s][i].get("dice") is not None]
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.asarray(out, float)


def vector_score(datos_modelo, ids, semillas, clave="prob_masa"):
    out = []
    for i in ids:
        vals = [datos_modelo[s][i][clave] for s in semillas
                if i in datos_modelo.get(s, {})
                and datos_modelo[s][i].get(clave) is not None]
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.asarray(out, float)


def correctos(datos_modelo, ids_pos, ids_neg, semillas):
    """Vector booleano de acierto de detección por imagen (120)."""
    out = []
    for i in ids_pos:
        vals = [datos_modelo[s][i]["pred_any"] for s in semillas
                if i in datos_modelo.get(s, {})]
        out.append(bool(np.mean(vals) >= 0.5) if vals else False)
    for i in ids_neg:
        vals = [not datos_modelo[s][i]["pred_any"] for s in semillas
                if i in datos_modelo.get(s, {})]
        out.append(bool(np.mean(vals) >= 0.5) if vals else False)
    return np.asarray(out, bool)


def parametros_millones(modelos):
    salida = {}
    for m in modelos:
        try:
            from modelos_tb import crear_modelo
            net, _ = crear_modelo(m, in_channels=1, out_channels=1,
                                  verbose=False)
            salida[m] = round(sum(p.numel() for p in net.parameters()) / 1e6, 3)
        except Exception:
            salida[m] = None
    return salida


# ============================================================================
# PRINCIPAL
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Análisis estadístico del benchmark.")
    ap.add_argument("--resultados", default=RES_DIR_DEF)
    ap.add_argument("--splits-dir", default=SPLITS_DEF)
    ap.add_argument("--salida", default=None,
                    help="Directorio de salida (por defecto resultados/analisis)")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--excluir", nargs="*", default=None,
                    help="Modelos a excluir del análisis (se declara en el informe)")
    ap.add_argument("--sin-parametros", action="store_true",
                    help="No instanciar modelos para contar parámetros")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    salida = a.salida or os.path.join(a.resultados, "analisis")
    os.makedirs(salida, exist_ok=True)

    modelos_bench = [m for m in ORDEN_BENCHMARK
                     if MODELOS[m].get("entrenable", False)]
    excluidos = list(a.excluir or [])
    modelos_bench = [m for m in modelos_bench if m not in excluidos]

    # Se cargan también los baselines entrenados (unet_fuerte) para la Tabla A,
    # pero NO entran al Friedman (preregistro §3).
    modelos_carga = modelos_bench + [b for b in ORDEN_BASELINES
                                     if MODELOS.get(b, {}).get("entrenable")
                                     and b not in excluidos]

    datos, ids_pos, ids_neg, faltantes = cargar_predicciones(
        a.resultados, modelos_carga, SEMILLAS, a.splits_dir)
    modelos = [m for m in modelos_carga if datos.get(m) and
               any(datos[m].get(s) for s in SEMILLAS)]
    if faltantes:
        print("Runs sin por_imagen_test.csv:")
        for m, ss in faltantes.items():
            print(f"   {m}: semillas {ss}")
    print(f"Modelos con test: {modelos}")

    # ---------------------------------------------------------- baselines
    baselines = calcular_baselines(a.splits_dir)
    for b in ("empty_mask", "bbox_mask"):
        if b in baselines and baselines[b]:
            datos[b] = {SEMILLAS[0]: baselines[b]}
            if b not in modelos:
                modelos.append(b)

    # ------------------------------------------------------- validar matriz
    ids_ok = ids_pos
    matriz = {}
    for m in modelos_bench:
        if not datos.get(m):
            continue
        v = vector_dice(datos[m], ids_ok, SEMILLAS)
        if np.isnan(v).any():
            print(f"   ⚠ {m}: {int(np.isnan(v).sum())} imágenes sin Dice; "
                  f"se excluye del Friedman (se reporta en Tabla A)")
            continue
        matriz[m] = v
    modelos_friedman = list(matriz)
    k = len(modelos_friedman)
    if k < 3:
        raise SystemExit("Se necesitan >= 3 modelos completos para Friedman.")

    # ----------------------------------------------------------- Tabla A
    params = {} if a.sin_parametros else parametros_millones(modelos)
    filas_a = []
    apendice = []
    for m in modelos:
        dm = datos[m]
        dice_m, dice_sd, dice_runs = media_por_semilla(dm, ids_pos, SEMILLAS, "dice")
        iou_m, iou_sd, _ = media_por_semilla(dm, ids_pos, SEMILLAS, "iou")
        hd_m, hd_sd, _ = media_por_semilla(dm, ids_pos, SEMILLAS, "hd95")
        as_m, as_sd, _ = media_por_semilla(dm, ids_pos, SEMILLAS, "assd")
        n_hd = contar_definido(dm, ids_pos, SEMILLAS, "hd95")
        n_as = contar_definido(dm, ids_pos, SEMILLAS, "assd")

        # segundos por época y Dice de val
        seg_ep, dice_val_runs = [], []
        for s in SEMILLAS:
            rjson = os.path.join(a.resultados, m, f"seed_{s}", "resultado.json")
            if not os.path.exists(rjson):
                continue
            with open(rjson, encoding="utf-8") as f:
                r = json.load(f)
            if r.get("dice_val") is not None:
                dice_val_runs.append(float(r["dice_val"]))
            if r.get("segundos") and r.get("epochs_entrenadas"):
                seg_ep.append(float(r["segundos"]) / float(r["epochs_entrenadas"]))
        dv_m = float(np.mean(dice_val_runs)) if dice_val_runs else None
        dv_sd = float(np.std(dice_val_runs)) if dice_val_runs else None
        sep_m = float(np.mean(seg_ep)) if seg_ep else None

        # AUC de detección por semilla
        aucs = []
        for s in SEMILLAS:
            filas = dm.get(s, {})
            if not filas or any(i not in filas for i in ids_pos + ids_neg):
                continue
            r = auc_de_long([filas[i].get("prob_masa") for i in ids_pos],
                            [filas[i].get("prob_masa") for i in ids_neg])
            if r:
                aucs.append(r["auc"])
        auc_m = float(np.mean(aucs)) if aucs else None
        auc_sd = float(np.std(aucs)) if aucs else None

        filas_a.append({
            "modelo": m,
            "dice_media": dice_m, "dice_de": dice_sd,
            "iou_media": iou_m,
            "hd95_media": hd_m, "hd95_n": n_hd,
            "assd_media": as_m, "assd_n": n_as,
            "parametros_M": params.get(m, None),
            "seg_epoca": None if sep_m is None else round(sep_m, 2),
            "dice_val_media": None if dv_m is None else round(dv_m, 6),
            "dice_val_de": None if dv_sd is None else round(dv_sd, 6),
            "auc_deteccion": None if auc_m is None else round(auc_m, 6),
            "auc_deteccion_de": None if auc_sd is None else round(auc_sd, 6),
            "n_semillas": int(np.sum(~np.isnan(dice_runs))),
            "nota": NOTAS.get(m, ""),
        })
        for s, val in zip(SEMILLAS, dice_runs):
            apendice.append({"modelo": m, "semilla": s,
                             "dice_test_media_imagen": None if np.isnan(val)
                             else round(float(val), 6)})

    # ----------------------------------------------------------- Tabla B
    M = np.vstack([matriz[m] for m in modelos_friedman])
    fri = friedman_iman(M)
    rangos = rangos_medios(M)
    orden = np.argsort(rangos)
    tabla_b = []
    for pos, idx in enumerate(orden, 1):
        tabla_b.append({"puesto": pos,
                        "modelo": modelos_friedman[idx],
                        "rango_medio": round(float(rangos[idx]), 4),
                        "dice_medio": round(float(np.mean(M[idx])), 6)})
    mejor = modelos_friedman[int(orden[0])]

    # ----------------------------------------------------------- Tabla C
    d_mejor = matriz[mejor]
    filas_c = []
    pvals = []
    tmp = []
    for m in modelos_friedman:
        if m == mejor:
            continue
        d = matriz[m] - d_mejor
        theta, lo, hi = bca_ic(d, a.n_bootstrap, a.seed)
        w = wilcoxon_vs(d)
        pvals.append(w["p"])
        tmp.append((m, d, theta, lo, hi, w))

    padj = holm(pvals) if pvals else []
    for (m, d, theta, lo, hi, w), pa in zip(tmp, padj):
        equiv = (lo is not None and hi is not None
                 and lo >= -DELTA and hi <= DELTA)
        if m == mejor:
            decision = "mejor"
        elif equiv:
            decision = "equivalente"
        elif pa < ALPHA and theta < 0:
            decision = "peor"
        elif pa < ALPHA and theta > 0:
            decision = "mejor"
        else:
            decision = "sin diferencia significativa"
        filas_c.append({
            "modelo": m,
            "comparado_con": mejor,
            "delta_dice": round(float(theta), 6),
            "ic95_bca_inf": round(float(lo), 6),
            "ic95_bca_sup": round(float(hi), 6),
            "p_wilcoxon": round(float(w["p"]), 6),
            "p_holm": round(float(pa), 6),
            "efecto_r": round(float(w["r"]), 4),
            "delta_relevancia": DELTA,
            "decision": decision,
        })

    # ----------------------------------------------------------- H1.c
    filas_d = []
    aucs = {}
    for m in modelos:
        dm = datos[m]
        if not dm.get(SEMILLAS[0]):
            continue
        if any(i not in dm[SEMILLAS[0]] or
               dm[SEMILLAS[0]][i].get("prob_masa") is None
               for i in ids_pos + ids_neg):
            continue
        score_pos = vector_score(dm, ids_pos, SEMILLAS)
        score_neg = vector_score(dm, ids_neg, SEMILLAS)
        aucs[m] = auc_de_long(score_pos, score_neg)

    comparaciones = []
    if mejor in aucs and aucs[mejor] is not None:
        p_delong = []
        tmp_d = []
        for m in modelos_friedman:
            if m == mejor or m not in aucs:
                continue
            r = delong_pareado(aucs[m], aucs[mejor])
            if r is None:
                continue
            p_delong.append(r["p"])
            tmp_d.append((m, r))
        padj_d = holm(p_delong) if p_delong else []
        for (m, r), pa in zip(tmp_d, padj_d):
            comparaciones.append({
                "tipo": "DeLong", "modelo": m, "vs": mejor,
                "auc_modelo": round(aucs[m]["auc"], 6),
                "auc_mejor": round(aucs[mejor]["auc"], 6),
                "estadistico_z": round(r["z"], 4),
                "p": round(r["p"], 6), "p_holm": round(float(pa), 6),
            })

    corr = {m: correctos(datos[m], ids_pos, ids_neg, SEMILLAS)
            for m in modelos if datos.get(m) and datos[m].get(SEMILLAS[0])}
    if mejor in corr:
        p_mcn = []
        tmp_m = []
        for m in modelos_friedman:
            if m == mejor or m not in corr:
                continue
            r = mcnemar_exacto(corr[m], corr[mejor])
            p_mcn.append(r["p"])
            tmp_m.append((m, r))
        padj_m = holm(p_mcn) if p_mcn else []
        for (m, r), pa in zip(tmp_m, padj_m):
            comparaciones.append({
                "tipo": "McNemar", "modelo": m, "vs": mejor,
                "aciertos_modelo": int(corr[m].sum()),
                "aciertos_mejor": int(corr[mejor].sum()),
                "n10": r["n10"], "n01": r["n01"],
                "estadistico_z": None,
                "p": round(r["p"], 6), "p_holm": round(float(pa), 6),
            })
    filas_d = comparaciones

    # ------------------------------------------- sensibilidad por semilla
    sens = []
    for s in SEMILLAS:
        Mm = []
        ok = True
        for m in modelos_friedman:
            v = vector_dice(datos[m], ids_pos, [s])
            if np.isnan(v).any():
                ok = False
                break
            Mm.append(v)
        if not ok:
            continue
        r = friedman_iman(np.vstack(Mm))
        rg = rangos_medios(np.vstack(Mm))
        sens.append({"semilla": s, "chi2": round(r["chi2"], 4),
                     "p_friedman": round(r["p_friedman"], 6),
                     "F_iman": None if r["F_iman"] is None
                     else round(r["F_iman"], 4),
                     "p_iman": None if r["p_iman"] is None
                     else round(r["p_iman"], 6),
                     "orden": " > ".join(
                         modelos_friedman[i] for i in np.argsort(rg))})

    # --------------------------------------------------- duplicados
    dup = []
    for i, m1 in enumerate(modelos_friedman):
        for m2 in modelos_friedman[i + 1:]:
            if np.allclose(matriz[m1], matriz[m2], atol=1e-12):
                dup.append((m1, m2))

    # ----------------------------------------------------------- salida
    def _csv(nombre, filas):
        if not filas:
            return
        # Unión de claves en orden de aparición: DeLong y McNemar comparten
        # tabla pero no todas las columnas (aciertos_*, n10/n01). Usar solo
        # las claves de la primera fila haría fallar DictWriter con las demás.
        campos = []
        for fila in filas:
            for k in fila:
                if k not in campos:
                    campos.append(k)
        ruta = os.path.join(salida, nombre)
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos, restval="")
            w.writeheader()
            w.writerows(filas)

    _csv("Tabla_A.csv", filas_a)
    _csv("Tabla_B.csv", tabla_b)
    _csv("Tabla_C.csv", filas_c)
    _csv("Tabla_D_deteccion.csv", filas_d)
    _csv("sensibilidad_friedman.csv", sens)
    _csv("apendice_runs.csv", apendice)

    json_out = {
        "generado_en": __import__("datetime").datetime.now().isoformat(
            timespec="seconds"),
        "delta": DELTA, "alpha": ALPHA, "n_bootstrap": a.n_bootstrap,
        "semilla_bootstrap": a.seed,
        "modelos": modelos,
        "modelos_friedman": modelos_friedman,
        "excluidos": excluidos,
        "mejor_modelo": mejor,
        "friedman": fri,
        "ranking": tabla_b,
        "tabla_c": filas_c,
        "deteccion": filas_d,
        "sensibilidad_por_semilla": sens,
        "duplicados": dup,
        "faltantes": faltantes,
        "n_test_pos": len(ids_pos), "n_test_neg": len(ids_neg),
    }
    with open(os.path.join(salida, "analisis_estadistico.json"), "w",
              encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------- informe MD
    L = []
    L.append("# Informe estadístico — Entregable 2\n")
    L.append(f"- **Generado:** {json_out['generado_en']}")
    L.append(f"- **Test primario:** {len(ids_pos)} imágenes (GT plata D1-R1)")
    L.append(f"- **Test secundario:** {len(ids_neg)} negativas (máscara vacía)")
    L.append(f"- **Modelos en Friedman:** {k} "
             f"({', '.join(modelos_friedman)})")
    L.append(f"- **Mejor modelo:** **{mejor}** (menor rango medio)\n")
    if excluidos:
        L.append(f"- **Excluidos por el usuario:** {', '.join(excluidos)}\n")
    if dup:
        L.append("> ⚠ **Predicciones idénticas detectadas:** " +
                 "; ".join(f"{a} = {b}" for a, b in dup) +
                 ". Corregir antes de publicar el ranking.\n")

    L.append("## Tabla B — Friedman\n")
    f_iman = ("—" if fri["F_iman"] is None
              else f"{fri['F_iman']:.4f}")
    p_iman = ("—" if fri["p_iman"] is None
              else f"{fri['p_iman']:.3g}")
    L.append(f"χ²_F = {fri['chi2']:.4f}, p = {fri['p_friedman']:.3g}; "
             f"F_F (Iman-Davenport) = {f_iman}, p = {p_iman}\n")
    L.append("| Puesto | Modelo | Rango medio | Dice medio |")
    L.append("|---:|---|---:|---:|")
    for f in tabla_b:
        L.append(f"| {f['puesto']} | {f['modelo']} | {f['rango_medio']:.3f} | "
                 f"{f['dice_medio']:.4f} |")
    L.append("")

    L.append("## Tabla C — Comparaciones vs el mejor\n")
    L.append("| Modelo | Δ Dice | IC 95% BCa | p Wilcoxon | p Holm | r | Decisión |")
    L.append("|---|---:|---|---:|---:|---:|---|")
    for f in filas_c:
        L.append(f"| {f['modelo']} | {f['delta_dice']:+.4f} | "
                 f"[{f['ic95_bca_inf']:+.4f}, {f['ic95_bca_sup']:+.4f}] | "
                 f"{f['p_wilcoxon']:.3g} | {f['p_holm']:.3g} | "
                 f"{f['efecto_r']:+.3f} | {f['decision']} |")
    L.append("")

    L.append("## Tabla A — Descriptiva\n")
    L.append("| Modelo | Dice test | IoU | HD95 (n) | ASSD (n) | Params (M) | "
             "s/época | Dice val | AUC det. | Nota |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for f in filas_a:
        def fmt(x, d=4):
            return "—" if x is None else f"{x:.{d}f}"
        L.append(f"| {f['modelo']} | {fmt(f['dice_media'])} ± "
                 f"{fmt(f['dice_de'])} | {fmt(f['iou_media'])} | "
                 f"{fmt(f['hd95_media'],2)} ({f['hd95_n']}) | "
                 f"{fmt(f['assd_media'],2)} ({f['assd_n']}) | "
                 f"{fmt(f['parametros_M'],2)} | {fmt(f['seg_epoca'],1)} | "
                 f"{fmt(f['dice_val_media'])} | {fmt(f['auc_deteccion'])} | "
                 f"{f.get('nota', '')} |")
    L.append("")

    if filas_d:
        L.append("## H1.c — Detección (DeLong / McNemar, Holm)\n")
        L.append("| Prueba | Modelo | vs | AUC / aciertos | z | p | p Holm |")
        L.append("|---|---|---|---:|---:|---:|---:|")
        for f in filas_d:
            z = f["estadistico_z"]
            z_txt = "—" if z is None else f"{z:.3f}"
            if f["tipo"] == "DeLong":
                extra = f"{f['auc_modelo']:.3f} vs {f['auc_mejor']:.3f}"
            else:
                extra = (f"{f['aciertos_modelo']} vs {f['aciertos_mejor']} "
                         f"(n10={f['n10']}, n01={f['n01']})")
            L.append(f"| {f['tipo']} | {f['modelo']} | {f['vs']} | {extra} | "
                     f"{z_txt} | {f['p']:.3g} | {f['p_holm']:.3g} |")
        L.append("")

    if sens:
        L.append("## Sensibilidad — Friedman por semilla\n")
        L.append("| Semilla | χ² | p | Orden |")
        L.append("|---:|---:|---:|---|")
        for f in sens:
            L.append(f"| {f['semilla']} | {f['chi2']} | {f['p_friedman']:.3g} | "
                     f"{f['orden']} |")
        L.append("")

    with open(os.path.join(salida, "INFORME_ESTADISTICO.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))

    # ----------------------------------------------------------- consola
    print("\n" + "=" * 74)
    print(f"Mejor modelo: {mejor}")
    print(f"Friedman chi2={fri['chi2']:.3f} p={fri['p_friedman']:.3g} | "
          f"Iman-Davenport F={fri['F_iman']} p={fri['p_iman']}")
    print(f"Salida: {salida}")
    if dup:
        print("⚠ Duplicados exactos:", dup)
    print("=" * 74)


if __name__ == "__main__":
    main()
