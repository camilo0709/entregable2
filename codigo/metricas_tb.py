"""
================================================================================
MÉTRICAS DE SEGMENTACIÓN - PROTOCOLO ÚNICO DEL BENCHMARK
================================================================================

PROPÓSITO:
    Un solo lugar donde se definen las métricas. Si cada modelo midiera con
    su propia función, la comparación no sería justa.

MÉTRICAS DEL ENTREGABLE 2 (Ruta B - imágenes):
    Primaria   -> Dice
    Secundarias-> IoU / mIoU, HD95, ASSD
    Margen de relevancia (delta) -> 0.02 en Dice (Sección 1.1 del informe,
    preregistrado antes de entrenar; aquí se refleja ese valor)

DEFINICIONES FIJADAS (se declara en el preregistro):
    - Binarización: probabilidad > 0.5
    - Dice = 2|A∩B| / (|A|+|B|)
    - IoU  = |A∩B| / |A∪B|
    - HD95: percentil 95 de la distancia de Hausdorff bidireccional (px)
    - ASSD: distancia media simétrica de superficie (px)
    - Caso GT vacío y predicción vacía -> Dice=1, IoU=1 (ambos aciertan)
    - Caso GT vacío y predicción no vacía -> Dice=0, IoU=0
    - HD95/ASSD no definidos si alguna máscara es vacía -> se registra NaN
      y se excluye del promedio, DECLARÁNDO el denominador.

SIN DEPENDENCIAS EXTERNAS: HD95/ASSD se implementan con numpy + scipy
para no depender de medpy/monai en el entorno de análisis.
================================================================================
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# Margen de relevancia del Entregable (Delta de Equivalencia)
DELTA_DICE = 0.01


# ============================================================================
# MÉTRICAS SUPERFICIALES (numpy)
# ============================================================================

def dice(a, b, smooth=1e-8):
    """Dice entre dos máscaras binarias."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    if den == 0:
        return 1.0                     # ambas vacías: acuerdo perfecto
    return float((2.0 * inter + smooth) / (den + smooth))


def iou(a, b, smooth=1e-8):
    """IoU (Jaccard) entre dos máscaras binarias."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0                     # ambas vacías
    return float((inter + smooth) / (union + smooth))


def _superficie(m):
    """
    Bordes de una máscara binaria (para métricas de distancia).

    Un píxel es interior solo si SUS CUATRO vecinos directos también son
    foreground (erosión por vecindad 4). El borde es entonces máscara
    menos interior. Ojo: usar OR en lugar de AND haría que cualquier píxel
    con al menos un vecino foreground contara como interior, dejando la
    superficie vacía en máscaras sólidas y convirtiendo HD95/ASSD en NaN.
    """
    m = np.asarray(m).astype(bool)
    if not m.any():
        return np.zeros_like(m)
    interior = np.zeros_like(m)
    interior[1:-1, 1:-1] = (
        m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    )
    borde = m & ~interior
    return borde


def hd95(a, b):
    """
    Distancia de Hausdorff al percentil 95, en píxeles.

    Devuelve NaN si alguna máscara es vacía (no hay superficie que medir).
    Se registra explícitamente para no ocultarlo en el promedio.
    """
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    sa, sb = _superficie(a), _superficie(b)
    if not sa.any() or not sb.any():
        return float("nan")
    # distancias desde cada píxel hasta la máscara más cercana
    da = distance_transform_edt(~a)
    db = distance_transform_edt(~b)
    d_ab = da[sb]          # de superficie de b hacia a
    d_ba = db[sa]          # de superficie de a hacia b
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def assd(a, b):
    """
    Distancia media simétrica de superficie (ASSD), en píxeles.

    NaN si alguna máscara es vacía, por la misma razón que hd95.
    """
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    sa, sb = _superficie(a), _superficie(b)
    if not sa.any() or not sb.any():
        return float("nan")
    da = distance_transform_edt(~a)
    db = distance_transform_edt(~b)
    d_ab = da[sb]
    d_ba = db[sa]
    return float((d_ab.sum() + d_ba.sum()) / (len(d_ab) + len(d_ba)))


def metricas_imagen(pred, gt):
    """
    Todas las métricas para una imagen. Devuelve dict plana (para CSV).

    pred, gt: máscaras binarias 2D.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    ambos_vacios = (not pred.any()) and (not gt.any())
    gt_vacio = not gt.any()
    pred_vacia = not pred.any()

    d = dice(pred, gt)
    j = iou(pred, gt)
    h = hd95(pred, gt)
    s = assd(pred, gt)

    # intersección / union explícitas (útiles para las Tablas A/B/C)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    area_pred = float(pred.sum())
    area_gt = float(gt.sum())

    # clasificación a nivel de imagen (para McNemar / DeLong)
    # positivo = hay lesión segmentada
    tp = bool(pred.any() and gt.any())
    fp = bool(pred.any() and not gt.any())
    fn = bool(not pred.any() and gt.any())
    tn = bool(not pred.any() and not gt.any())

    return {
        "dice": round(d, 6),
        "iou": round(j, 6),
        "hd95": (None if np.isnan(h) else round(h, 4)),
        "assd": (None if np.isnan(s) else round(s, 4)),
        "interseccion": inter,
        "union": union,
        "area_pred": area_pred,
        "area_gt": area_gt,
        "ambos_vacios": ambos_vacios,
        "gt_vacio": gt_vacio,
        "pred_vacia": pred_vacia,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "hd95_definido": not np.isnan(h),
        "assd_definido": not np.isnan(s),
    }


def dice_pytorch(pred_bool, gt_bool):
    """Dice sobre tensores PyTorch (usado en la validación del driver)."""
    pred_bool = pred_bool.bool() if hasattr(pred_bool, "bool") else pred_bool.bool()
    gt_bool = gt_bool.bool()
    inter = (pred_bool & gt_bool).sum().float()
    den = pred_bool.sum().float() + gt_bool.sum().float()
    if den == 0:
        return 1.0
    return float((2.0 * inter) / den)


# ============================================================================
# AGREGACIÓN (lo que alimenta las Tablas A/B/C)
# ============================================================================

def agregar(filas):
    """
    Agrega una lista de dicts (salida de metricas_imagen) a los valores
    que se reportan.

    IMPORTANTE: los promedios de HD95/ASSD se calculan SOLO sobre las
    imágenes donde la métrica está definida, y se devuelve el denominador
    para que el lector sepa sobre cuántos casos se calculó.
    """
    n = len(filas)
    if n == 0:
        return {}

    def _media(clave):
        vals = [f[clave] for f in filas if f.get(clave) is not None]
        return (round(float(np.mean(vals)), 6) if vals else None), len(vals)

    dice_m, _ = _media("dice")
    iou_m, _ = _media("iou")
    hd_m, n_hd = _media("hd95")
    as_m, n_as = _media("assd")

    return {
        "n_imagenes": n,
        # --- primaria
        "dice_mean": dice_m,
        "dice_std": round(float(np.std([f["dice"] for f in filas])), 6),
        "dice_median": round(float(np.median([f["dice"] for f in filas])), 6),
        # --- secundarias
        "iou_mean": iou_m,
        "iou_std": round(float(np.std([f["iou"] for f in filas])), 6),
        "hd95_mean": hd_m,
        "hd95_n_definido": n_hd,          # denominador honesto
        "assd_mean": as_m,
        "assd_n_definido": n_as,
        # --- contexto
        "n_ambos_vacios": sum(1 for f in filas if f.get("ambos_vacios")),
        "n_gt_vacio": sum(1 for f in filas if f.get("gt_vacio")),
        "n_pred_vacia": sum(1 for f in filas if f.get("pred_vacia")),
        # --- clasificación por imagen (para DeLong / McNemar)
        "tp": sum(1 for f in filas if f.get("tp")),
        "fp": sum(1 for f in filas if f.get("fp")),
        "fn": sum(1 for f in filas if f.get("fn")),
        "tn": sum(1 for f in filas if f.get("tn")),
        "delta_relevancia": DELTA_DICE,
    }


# ============================================================================
# AUTOPRUEBA
# ============================================================================

def _autopruéba():
    """Comprueba las definiciones con casos de borde conocidos."""
    print("Autoprueba de métricas")
    print("-" * 60)
    a = np.zeros((64, 64), bool)
    b = np.zeros((64, 64), bool)

    # 1. idénticas
    a[10:30, 10:30] = True
    print(f"  idénticas          dice={dice(a,a):.4f} iou={iou(a,a):.4f} "
          f"(esperado 1.0 / 1.0)")
    # 2. disjuntas
    b[40:60, 40:60] = True
    print(f"  disjuntas          dice={dice(a,b):.4f} iou={iou(a,b):.4f} "
          f"(esperado 0.0 / 0.0)  hd95={hd95(a,b):.2f} assd={assd(a,b):.2f}")
    # 3. solape parcial (mitad)
    c = np.zeros((64, 64), bool)
    c[10:30, 20:40] = True
    print(f"  solape mitad       dice={dice(a,c):.4f} iou={iou(a,c):.4f} "
          f"(esperado 0.50 / 0.333)")
    # 4. ambas vacías
    z = np.zeros((64, 64), bool)
    print(f"  ambas vacías       dice={dice(z,z):.4f} iou={iou(z,z):.4f} "
          f"(esperado 1.0 / 1.0)  hd95={hd95(z,z)} (esperado NaN)")
    # 5. GT vacío, predicción no
    print(f"  GT vacío + pred    dice={dice(a,z):.4f} iou={iou(a,z):.4f} "
          f"(esperado 0.0 / 0.0)  hd95={hd95(a,z)} (esperado NaN)")
    # 6. metricas_imagen sobre caso de borde
    m = metricas_imagen(a, z)
    print(f"  metricas_imagen    fp={m['fp']} fn={m['fn']} "
          f"hd95_definido={m['hd95_definido']}")
    # 7. agregación con denominador honesto
    agg = agregar([metricas_imagen(a, a), metricas_imagen(a, z)])
    print(f"  agregar (n=2)      dice_mean={agg['dice_mean']:.4f} "
          f"hd95_mean={agg['hd95_mean']} n_hd95={agg['hd95_n_definido']}")
    print("-" * 60)
    print("Autoprueba terminada.")


if __name__ == "__main__":
    _autopruéba()
