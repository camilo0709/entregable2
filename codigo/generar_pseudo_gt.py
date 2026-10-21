"""
================================================================================
GENERADOR DE GT PLATA (PSEUDO-MASCARAS SAM 2) - REVISION D1-R1
================================================================================

PROPOSITO:
    Genera UNICA y HOMogeneamente todo el ground truth plata del Entregable 2
    con un pipeline congelado y versionado:

        PNG original TBX11K (512x512)
          -> bbox_mask  (rectangulos del anotador original, escala 512)
          -> SAM 2 con prompt de caja (1 mascara por caja, mejor score)
          -> OR entre cajas + limpieza de componentes pequenos
          -> mascara refinada

    Salidas:
      1) train/val  -> datos/npz_dataset/<split>/<id>.npz
                       (image, bbox_mask, mask, label, tb_type, fname, bbox_count)
      2) test_pos   -> datos/ground_truth_test/<id>.png  (0/255, 512x512)
                       [D1-R1: GT plata del test, declarado en
                        docs/DECISION_D1_GROUND_TRUTH.md seccion R1]

    Al terminar escribe datos/pseudo_gt_manifest_v2.json con hash del
    checkpoint, parametros, conteos y marca de tiempo: cualquier corrida
    futura se puede verificar contra ese manifiesto.

POR QUE REGENERAR TODO (decision documentada):
    Los npz heredados venian de una generacion previa desalineada con los
    CSV sellados (solo 43/194 pool A coincidian). Regenerar los 194 + 80 con
    UN solo script + UN solo checkpoint elimina la heterogeneidad de version
    y hace el GT plata reproducible desde cero.

DTERMINISMO:
    - SAM 2 en modo eval, sin gradiente, semilla fija (42)
    - Sin augmentation, sin TTA, sin votacion
    - Post-proceso determinista: se eliminan componentes conexos < MIN_PX

USO:
    python codigo/generar_pseudo_gt.py --limit 3        # smoke test
    python codigo/generar_pseudo_gt.py                  # todo (train+val+test)
    python codigo/generar_pseudo_gt.py --splits test_pos --gt-dir datos/ground_truth_test
    python codigo/generar_pseudo_gt.py --device cpu     # forzar CPU
================================================================================
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(ROOT, "datos", "splits")
NPZ_DIR = os.path.join(ROOT, "datos", "npz_dataset")
GT_DIR = os.path.join(ROOT, "datos", "ground_truth_test")
TBX_IMG = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K", "imgs")
TBX_TB = os.path.join(TBX_IMG, "tb")
TBX_HEALTH = os.path.join(TBX_IMG, "health")
TBX_JSON = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K",
                        "annotations", "json")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")

CKPT_NOMBRE = "sam2_hiera_tiny.pt"
# NB: sam2.0 vive en la carpeta 072824 (092824 solo tiene sam2.1_*)
CKPT_URL = ("https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_tiny.pt")
MODEL_CFG = "sam2_hiera_t.yaml"          # config del paquete sam2 oficial

SEMILLA = 42
MIN_PX = 32                # componentes conexos mas pequeños se eliminan
TAM = (512, 512)

# categorias COCO de TBX11K -> tb_type del proyecto
CAT_A_TIPO = {1: "active_tb", 2: "latent_tb"}   # 2 = Obsolete (= latent)


# ============================================================================
# UTILIDADES DE DATOS
# ============================================================================

def leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cargar_anotaciones():
    """Une los JSON COCO de TBX11K -> {id_imagen: [ {bbox, cat}, ... ]}."""
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
            # COCO: bbox = [x, y, w, h] en la resolucion original de la imagen
            clave = os.path.splitext(os.path.basename(im["file_name"]))[0]
            x, y, w, h = a["bbox"]
            mapa.setdefault(clave, []).append(
                {"bbox": (x, y, w, h), "cat": a.get("category_id", 1),
                 "img_w": im.get("width"), "img_h": im.get("height")})
    return mapa


def cajas_a_bbox_mask(cajas, escala, offset, tam=TAM):
    """Dibuja los rectangulos escalados. escala: factores (sx, sy)."""
    m = np.zeros(tam, dtype=np.uint8)
    for c in cajas:
        x, y, w, h = c["bbox"]
        x0 = int(round((x + offset[0]) * escala[0]))
        y0 = int(round((y + offset[1]) * escala[1]))
        x1 = int(round((x + w + offset[0]) * escala[0]))
        y1 = int(round((y + h + offset[1]) * escala[1]))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(tam[0], x1), min(tam[1], y1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1
    return m


def limpiar(mask, min_px=MIN_PX):
    """Elimina componentes conexos < min_px (artefactos). Determinista."""
    from scipy.ndimage import label
    lab, n = label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    tamanos = np.bincount(lab.ravel())
    keep = np.zeros_like(mask, dtype=bool)
    for k in range(1, n + 1):
        if tamanos[k] >= min_px:
            keep |= (lab == k)
    return keep.astype(np.uint8)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ============================================================================
# SAM 2 (se importa tarde: el smoke test sin SAM 2 puede listar pendientes)
# ============================================================================

_SAM = {"predictor": None, "ckpt": None}


def _asegurar_sam2(ckpt_ruta, device):
    if _SAM["predictor"] is not None and _SAM["ckpt"] == ckpt_ruta:
        return _SAM["predictor"]
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists(ckpt_ruta):
        raise FileNotFoundError(
            f"Falta el checkpoint de SAM 2: {ckpt_ruta}\n"
            f"  Descargalo con:  python codigo/generar_pseudo_gt.py --descargar-ckpt\n"
            f"  (o manualmente desde {CKPT_URL})")
    torch.manual_seed(SEMILLA)
    modelo = build_sam2(MODEL_CFG, ckpt_ruta, device=device)
    predictor = SAM2ImagePredictor(modelo)
    _SAM["predictor"] = predictor
    _SAM["ckpt"] = ckpt_ruta
    return predictor


def descargar_ckpt():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    destino = os.path.join(CHECKPOINTS_DIR, CKPT_NOMBRE)
    if os.path.exists(destino):
        print(f"Ya existe: {destino}")
        return destino
    print(f"Descargando {CKPT_URL} ...")
    import urllib.request
    urllib.request.urlretrieve(CKPT_URL, destino)
    print(f"Guardado: {destino} ({os.path.getsize(destino)/1e6:.1f} MB)")
    return destino


def predecir_sam2(predictor, img_u8, cajas, escala, offset):
    """
    set_image() una vez por imagen + una mascara por caja (multimask ->
    mejor score), OR entre cajas. El bbox original se escala a la
    imagen ya redimencionada. La imagen de entrada es escala de grises
    uint8; SAM 2 espera RGB uint8 HWC.
    """
    import torch
    img_rgb = np.stack([img_u8] * 3, axis=-1)         # (H,W,3) RGB
    predictor.set_image(img_rgb)
    resultados = []
    for c in cajas:
        x, y, w, h = c["bbox"]
        box = np.array([
            (x + offset[0]) * escala[0], (y + offset[1]) * escala[1],
            (x + w + offset[0]) * escala[0], (y + h + offset[1]) * escala[1]],
            dtype=np.float32)
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                box=box, multimask_output=True)
        mejor = int(np.argmax(scores))
        resultados.append(masks[mejor].astype(np.uint8))
    union = np.zeros(img_u8.shape, dtype=np.uint8)
    for r in resultados:
        union |= r
    return union


def visualizar(img_u8, bbox_mask, mask, salida):
    import cv2
    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[bbox_mask > 0] = (0, 165, 255)       # naranja = caja
    overlay[mask > 0] = (0, 255, 0)               # verde = refinada
    vis = cv2.addWeighted(base, 0.6, overlay, 0.4, 0)
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    for (x0, y0, w, h) in _contornos(bbox_mask):   # boundingRect = (x,y,w,h)
        cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (0, 165, 255), 1)
    cv2.imwrite(salida, vis)


def _contornos(m):
    import cv2
    cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cs]


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def generar(splits, gt_dir, ckpt, device, limit=None, viz=False,
            solo_listar=False, splits_dir=SPLITS_DIR, out_npz=None,
            manifiesto_ruta=None):
    import cv2
    from PIL import Image

    anot = cargar_anotaciones()
    pendientes = []          # (split, fila, imagen_ruta, cajas)
    for sp in splits:
        ruta_csv = os.path.join(splits_dir, f"{sp}.csv")
        filas = leer_csv(ruta_csv)
        for fila in filas:
            iid = fila["id"]
            if sp in ("train", "val"):
                # en train/val la columna 'ruta' apunta al NPZ; el PNG
                # original vive en imgs/tb/<id>.png (todos pool A son tb)
                img_ruta = os.path.join(TBX_TB, f"{iid}.png")
                if not os.path.exists(img_ruta):
                    img_ruta = os.path.join(TBX_HEALTH, f"{iid}.png")
            else:
                # test_*: el CSV trae la ruta PNG (remapeada a local)
                r = fila["ruta"].replace("\\", "/")
                k = r.find("datos/")
                img_ruta = os.path.join(ROOT, r[k:]) if k >= 0 else r
                if not os.path.exists(img_ruta):
                    img_ruta = os.path.join(TBX_TB, f"{iid}.png")
            cajas = anot.get(iid, [])
            pendientes.append((sp, fila, img_ruta, cajas))

    print("=" * 74)
    print("GENERADOR DE GT PLATA - SAM 2 (revision D1-R1)")
    print("=" * 74)
    print(f"  splits      : {splits}")
    print(f"  pendientes  : {len(pendientes)} imagenes")
    sin_caja = [p[1]['id'] for p in pendientes if not p[3]]
    sin_png = [p[1]['id'] for p in pendientes if not os.path.exists(p[2])]
    print(f"  sin anotacion bbox : {len(sin_caja)}"
          + (f" -> {sin_caja[:8]}" if sin_caja else ""))
    print(f"  sin PNG en disco  : {len(sin_png)}"
          + (f" -> {sin_png[:8]}" if sin_png else ""))
    if solo_listar:
        return {"pendientes": len(pendientes), "sin_caja": sin_caja,
                "sin_png": sin_png}
    if sin_png:
        raise SystemExit(
            f"Faltan {len(sin_png)} PNG. Descarga TBX11K completo primero "
            f"(ver docs/DECISION_D1_GROUND_TRUTH.md R1).")

    predictor = _asegurar_sam2(ckpt, device)
    t0 = time.time()
    stats = {"generadas": 0, "por_split": {}, "errores": []}

    for k, (sp, fila, img_ruta, cajas) in enumerate(pendientes, 1):
        if limit and k > limit:
            break
        iid = fila["id"]
        try:
            img = np.array(Image.open(img_ruta).convert("L"))
            h0, w0 = img.shape
            escala = (TAM[0] / w0, TAM[1] / h0)
            offset = (0.0, 0.0)
            if (w0, h0) != TAM:
                img = cv2.resize(img, TAM, interpolation=cv2.INTER_AREA)
            img_u8 = img.astype(np.uint8)

            bbox_mask = cajas_a_bbox_mask(cajas, escala, offset)
            if not cajas:
                raise ValueError("sin cajas bbox")
            mask = predecir_sam2(predictor, img_u8, cajas, escala, offset)
            mask = limpiar(mask, MIN_PX)
            if mask.sum() == 0:
                # anti-vacio: si SAM 2 devolvio nada, usar la caja
                # (test_pos exige no-vacio; se registra como fallback)
                mask = bbox_mask.copy()
                stats["errores"].append(f"{iid}: mask vacia -> fallback bbox")

            if sp in ("train", "val"):
                if out_npz:
                    # destino de prueba / staging explícito (smoke, Colab)
                    destino = os.path.join(out_npz, sp, f"{iid}.npz")
                else:
                    # la ruta destino la dicta el CSV sellado (remapeada a local)
                    wr = fila["ruta"].replace("\\", "/")
                    j = wr.find("datos/")
                    if j < 0:
                        raise ValueError(f"ruta del CSV sin segmento datos/: "
                                         f"{fila['ruta']}")
                    destino = os.path.join(ROOT, wr[j:])
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                np.savez_compressed(
                    destino,
                    image=img_u8,
                    bbox_mask=bbox_mask.astype(np.uint8),
                    # convenio del npz original: binario {0,1} (verificado
                    # contra datos/npz_dataset_v1_old)
                    mask=mask.astype(np.uint8),
                    label=np.int64(fila.get("label", 1) or 1),
                    tb_type=np.str_(fila.get("tb_type", "")),
                    fname=np.str_(iid + ".png"),
                    bbox_count=np.int64(len(cajas)))
            else:  # test_pos u otro split con GT en PNG
                os.makedirs(gt_dir, exist_ok=True)
                m255 = (mask > 0).astype(np.uint8) * 255
                Image.fromarray(m255).save(os.path.join(gt_dir, f"{iid}.png"))

            if viz:
                visualizar(img_u8, bbox_mask, mask,
                           os.path.join(ROOT, "datos", "npz_dataset",
                                        "visualizations_v2",
                                        f"viz_{k:03d}_{iid}.png"))

            stats["generadas"] += 1
            stats["por_split"][sp] = stats["por_split"].get(sp, 0) + 1
            if k % 25 == 0:
                print(f"  [{k}/{len(pendientes)}] {sp}:{iid} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            stats["errores"].append(f"{iid}: {type(e).__name__}: {e}")
            print(f"  [!!] {iid}: {e}", flush=True)

    # ---------------------------------------------------------- manifiesto
    ckpt_sha = sha256(ckpt) if os.path.exists(ckpt) else None
    manifiesto = {
        "version": "pseudo_gt_v2_d1r1",
        "generado": datetime.now().isoformat(timespec="seconds"),
        "script": "codigo/generar_pseudo_gt.py",
        "script_sha256": sha256(os.path.abspath(__file__)),
        "checkpoint": ckpt,
        "checkpoint_sha256": ckpt_sha,
        "model_cfg": MODEL_CFG,
        "semilla": SEMILLA,
        "min_component_px": MIN_PX,
        "device": device,
        "splits": splits,
        "stats": stats,
        "duracion_s": round(time.time() - t0, 1),
        "declaracion": ("GT plata (silver standard) generado con SAM 2 "
                        "externo a TBX11K. Ver docs/DECISION_D1_GROUND_TRUTH.md "
                        "seccion R1. Validacion humana de submuestra: "
                        "docs/PROTOCOLO_ANOTACION.md."),
    }
    ruta_man = manifiesto_ruta or os.path.join(
        ROOT, "datos", "pseudo_gt_manifest_v2.json")
    with open(ruta_man, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)
    print("-" * 74)
    print(f"Generadas: {stats['generadas']}  {stats['por_split']}")
    print(f"Errores  : {len(stats['errores'])}")
    for e in stats["errores"][:10]:
        print(f"   - {e}")
    print(f"Manifiesto: {ruta_man}")
    return manifiesto


def main():
    ap = argparse.ArgumentParser(description="Genera el GT plata con SAM 2 (D1-R1)")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test_pos"])
    ap.add_argument("--gt-dir", default=GT_DIR)
    ap.add_argument("--ckpt", default=os.path.join(CHECKPOINTS_DIR, CKPT_NOMBRE))
    ap.add_argument("--device", default=None, help="cpu / cuda (auto)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--listar", action="store_true", help="Solo listar pendientes")
    ap.add_argument("--descargar-ckpt", action="store_true")
    ap.add_argument("--splits-dir", default=SPLITS_DIR)
    ap.add_argument("--out-npz", default=None,
                    help="Escribe los npz aquí (staging/smoke) en vez de la "
                         "ruta dictada por los CSV sellados")
    ap.add_argument("--manifiesto", default=None,
                    help="Ruta alternativa del manifiesto de generación")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.descargar_ckpt:
        descargar_ckpt()
        return

    device = a.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"   # --listar no necesita torch

    generar(a.splits, a.gt_dir, a.ckpt, device, limit=a.limit,
            viz=a.viz, solo_listar=a.listar, splits_dir=a.splits_dir,
            out_npz=a.out_npz, manifiesto_ruta=a.manifiesto)


if __name__ == "__main__":
    main()
