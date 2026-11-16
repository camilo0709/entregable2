"""
================================================================================
EVALUACIÓN EN TEST - BENCHMARK ENTREGABLE 2
================================================================================

PROPÓSITO:
    Cargar el checkpoint de cada run (`resultados/<modelo>/seed_<n>/
    mejor_modelo.pt`) y evaluarlo UNA sola vez sobre el test sellado:

        test_pos  80 imágenes  GT plata SAM 2 congelado (D1-R1)
        test_neg  40 imágenes  máscara vacía por construcción

    Escribe, por run:
        por_imagen_test.csv   métricas por imagen (pos + neg, 120 filas)
    y actualiza `resultado.json` con los agregados de test, SIN tocar la
    validación (`dice_val`) ni el historial.

    Además mantiene `resultados/resumen_test.csv`, que es lo que consume
    `codigo/analisis_estadistico.py` para las Tablas A/B/C.

CONVENCIÓN (fijada en el preregistro):
    - Binarización de la probabilidad: > 0.5
    - Dice/IoU/HD95/ASSD: `codigo/metricas_tb.py` (una sola definición)
    - Score continuo de detección (DeLong): masa de probabilidad predicha
      = suma de sigmoid / 1000  (preregistro §6.4)
    - El test NO se usa para seleccionar época ni umbral: aquí solo se mide
      el checkpoint ya congelado por la mejor época de validación.

USO:
    python codigo/evaluar_test.py                    # todo lo pendiente
    python codigo/evaluar_test.py --modelos unet --semillas 42
    python codigo/evaluar_test.py --sobrescribir     # re-evaluar aunque exista
    python codigo/evaluar_test.py --dry-run

NOTA:
    nnU-Net no se evalúa aquí: su pipeline (`entrenamiento_nnunet.py`) ya
    escribe `por_imagen_test.csv` con los mismos campos.
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
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_tb import DatasetTB, SPLITS_DIR                      # noqa: E402
from modelos_tb import (MODELOS, ORDEN_BENCHMARK, ORDEN_BASELINES,  # noqa: E402
                        SEMILLAS, SEMILLAS_EXTRA, ORDEN_ABLACIONES, crear_modelo)
import metricas_tb as M                                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR_DEF = os.path.join(ROOT, "resultados")

UMBRAL = 0.5
COLS_TEST = [
    "split", "id", "tb_type", "n_boxes",
    "dice", "iou", "hd95", "assd",
    "interseccion", "union", "area_pred", "area_gt",
    "ambos_vacios", "gt_vacio", "pred_vacia",
    "tp", "fp", "fn", "tn", "hd95_definido", "assd_definido",
    "prob_masa", "pred_any",
]


# ============================================================================
# UTILIDADES
# ============================================================================

def ruta_run(res_dir, modelo, semilla):
    return os.path.join(res_dir, modelo, f"seed_{semilla}")


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def escribir_json(ruta, datos):
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ruta)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def run_evaluado(res_dir, modelo, semilla):
    """Un run está evaluado si su resultado.json trae dice_test y existe el
    CSV por imagen de test."""
    rd = ruta_run(res_dir, modelo, semilla)
    rjson = os.path.join(rd, "resultado.json")
    rcsv = os.path.join(rd, "por_imagen_test.csv")
    if not (os.path.exists(rjson) and os.path.exists(rcsv)):
        return False
    try:
        r = leer_json(rjson)
    except Exception:
        return False
    return r.get("estado") == "completado" and "dice_test" in r


# ============================================================================
# INFERENCIA
# ============================================================================

def _infos_a_lista(infos, n):
    """Colapsa el dict de infos que devuelve el DataLoader (listas por clave)."""
    return [{k: infos[k][i] for k in infos} for i in range(n)]


@torch.inference_mode()
def inferir_dataset(net, ds, device, batch_size=8, num_workers=0):
    """Devuelve (filas_metricas, probabilidades_por_imagen)."""
    net.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers)
    filas = []
    for xb, yb, infos in dl:
        xb = xb.to(device, non_blocking=True)
        prob = torch.sigmoid(net(xb))
        prob_np = prob[:, 0].float().cpu().numpy()
        gt_np = (yb[:, 0].numpy() > 0.5).astype(np.uint8)
        info_list = _infos_a_lista(infos, xb.size(0))
        for k in range(xb.size(0)):
            pred = (prob_np[k] > UMBRAL).astype(np.uint8)
            m = M.metricas_imagen(pred, gt_np[k])
            m["id"] = info_list[k]["id"]
            m["tb_type"] = info_list[k].get("tb_type", "")
            m["n_boxes"] = info_list[k].get("n_boxes", 0)
            m["prob_masa"] = round(float(prob_np[k].sum()) / 1000.0, 6)
            m["pred_any"] = bool(pred.any())
            filas.append(m)
    return filas


def inferir_split(net, split, splits_dir, device, batch_size, num_workers):
    """Inferencia de un split con reducción automática de batch si hay OOM
    (RTX 3050 de 6 GB no aguanta batch 8 en UNETR/DeepLabV3+ a 512²)."""
    while True:
        ds = DatasetTB(split, splits_dir=splits_dir, augmentar=False,
                       verbose=False)
        try:
            return inferir_dataset(net, ds, device, batch_size, num_workers)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            oom = isinstance(e, torch.cuda.OutOfMemoryError) or \
                "out of memory" in str(e).lower()
            if not oom or batch_size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"   OOM en {split}: reintento con batch={batch_size}",
                  flush=True)


def evaluar_run(modelo, semilla, res_dir, device, splits_dir,
                batch_size, num_workers):
    rd = ruta_run(res_dir, modelo, semilla)
    rjson = os.path.join(rd, "resultado.json")
    ckpt = os.path.join(rd, "mejor_modelo.pt")

    if not os.path.exists(rjson):
        return {"estado": "sin_resultado", "modelo": modelo, "semilla": semilla}
    r = leer_json(rjson)
    if r.get("estado") != "completado":
        return {"estado": f"run_{r.get('estado')}", "modelo": modelo,
                "semilla": semilla}
    if not os.path.exists(ckpt):
        return {"estado": "sin_checkpoint", "modelo": modelo, "semilla": semilla}

    # el checkpoint guarda la config usada (incl. hiperparámetros de HPO)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net, meta = crear_modelo(
        modelo, in_channels=1, out_channels=1, verbose=False,
        params_extra=(ck.get("config") or {}).get("params_modelo"))
    estado = ck.get("state_dict", ck)
    net.load_state_dict(estado)
    net = net.to(device)
    net.eval()

    filas = []
    for split in ("test_pos", "test_neg"):
        f = inferir_split(net, split, splits_dir, device, batch_size,
                          num_workers)
        for fila in f:
            fila["split"] = split
        filas.extend(f)

    # ------------------------------------------------------------- CSV test
    os.makedirs(rd, exist_ok=True)
    rcsv = os.path.join(rd, "por_imagen_test.csv")
    with open(rcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS_TEST)
        w.writeheader()
        for fila in filas:
            w.writerow({c: fila.get(c) for c in COLS_TEST})

    # --------------------------------------------------------- agregados
    pos = [x for x in filas if x["split"] == "test_pos"]
    neg = [x for x in filas if x["split"] == "test_neg"]

    def _media(filas_, clave):
        vals = [f[clave] for f in filas_ if f.get(clave) is not None]
        return (round(float(np.mean(vals)), 6) if vals else None), len(vals)

    dice_pos, _ = _media(pos, "dice")
    iou_pos, _ = _media(pos, "iou")
    hd95_pos, n_hd = _media(pos, "hd95")
    assd_pos, n_as = _media(pos, "assd")

    tp = sum(1 for x in pos if x["tp"])
    fn = sum(1 for x in pos if x["fn"])
    tn = sum(1 for x in neg if x["tn"])
    fp = sum(1 for x in neg if x["fp"])
    sens = round(tp / (tp + fn), 6) if (tp + fn) else None
    spec = round(tn / (tn + fp), 6) if (tn + fp) else None

    resumen = {
        "estado": "completado",
        "modelo": modelo,
        "semilla": semilla,
        "dice_val": r.get("dice_val"),
        "dice_test": dice_pos,
        "dice_test_std_imagen": round(
            float(np.std([x["dice"] for x in pos])), 6) if pos else None,
        "iou_test": iou_pos,
        "hd95_test": hd95_pos,
        "hd95_n_definido": n_hd,
        "assd_test": assd_pos,
        "assd_n_definido": n_as,
        "n_test_pos": len(pos),
        "n_test_neg": len(neg),
        "sensibilidad": sens,
        "especificidad": spec,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "umbral": UMBRAL,
        "gt_test": "pseudo_sam2_d1r1 (plata congelada)",
        "sha256_checkpoint": sha256(ckpt),
        "evaluado_en": datetime.now().isoformat(timespec="seconds"),
        "segundos_eval": None,
    }
    return resumen


# ============================================================================
# RESUMEN AGREGADO
# ============================================================================

RESUMEN_COLS = [
    "modelo", "semilla", "dice_val", "dice_test", "dice_test_std_imagen",
    "iou_test", "hd95_test", "hd95_n_definido", "assd_test",
    "assd_n_definido", "n_test_pos", "n_test_neg",
    "sensibilidad", "especificidad", "tp", "fn", "tn", "fp",
    "sha256_checkpoint", "evaluado_en",
]


def escribir_resumen_test(res_dir, modelos, semillas):
    """Reconstruye resumen_test.csv desde los resultado.json individuales."""
    filas = []
    for m in modelos:
        for s in semillas:
            rjson = os.path.join(ruta_run(res_dir, m, s), "resultado.json")
            if not os.path.exists(rjson):
                continue
            try:
                r = leer_json(rjson)
            except Exception:
                continue
            if "dice_test" not in r:
                continue
            filas.append({c: r.get(c) for c in RESUMEN_COLS})
    if not filas:
        return []
    ruta = os.path.join(res_dir, "resumen_test.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESUMEN_COLS)
        w.writeheader()
        w.writerows(filas)
    return filas


# ============================================================================
# PRINCIPAL
# ============================================================================

def verificar_checkpoints(modelos, semillas, res_dir):
    """Carga cada checkpoint en su arquitectura SIN inferir.

    Sirve para detectar de antemano diferencias de versión de smp/monai
    (state_dict incompatible) antes de lanzar horas de evaluación.
    """
    import gc
    ok, malos = [], []
    for m in modelos:
        for s in semillas:
            ckpt = os.path.join(ruta_run(res_dir, m, s), "mejor_modelo.pt")
            if not os.path.exists(ckpt):
                malos.append((m, s, "sin checkpoint"))
                continue
            try:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                net, _ = crear_modelo(
                    m, in_channels=1, out_channels=1, verbose=False,
                    params_extra=(ck.get("config") or {}).get("params_modelo"))
                estado = ck.get("state_dict", ck)
                net.load_state_dict(estado)
                ok.append((m, s))
            except Exception as e:
                malos.append((m, s, f"{type(e).__name__}: {e}"))
            finally:
                net = None
                gc.collect()
    print(f"Checkpoints cargados: {len(ok)}/{len(ok) + len(malos)}")
    if malos:
        print("INCOMPATIBLES:")
        for m, s, e in malos:
            print(f"   {m}/{s}: {e}")
    return ok, malos


def main():
    ap = argparse.ArgumentParser(
        description="Evalúa los checkpoints del benchmark sobre el test sellado.")
    ap.add_argument("--resultados", default=RES_DIR_DEF)
    ap.add_argument("--splits-dir", default=SPLITS_DIR)
    ap.add_argument("--modelos", nargs="*", default=None,
                    help="Claves a evaluar (por defecto: benchmark sin nnunet)")
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--sobrescribir", action="store_true",
                    help="Re-evalúa aunque dice_test ya exista")
    ap.add_argument("--verificar", action="store_true",
                    help="Solo comprueba que cada checkpoint carga en su "
                         "arquitectura (sin inferir)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if a.device == "auto" \
        else torch.device(a.device)

    modelos = a.modelos or (
        [m for m in ORDEN_BENCHMARK
         if MODELOS[m].get("entrenable", False)
         and MODELOS[m].get("backend") != "nnunet"]
        + [b for b in ORDEN_BASELINES
           if MODELOS.get(b, {}).get("entrenable", False)]
        + [c for c in ["lacca_unet"] + list(ORDEN_ABLACIONES)
           if MODELOS.get(c, {}).get("entrenable", False)]
    )
    semillas = a.semillas or (SEMILLAS + SEMILLAS_EXTRA)

    print("=" * 74)
    print("EVALUACIÓN EN TEST - ENTREGABLE 2")
    print("=" * 74)
    print(f"Device      : {device}")
    print(f"Resultados  : {a.resultados}")
    print(f"Modelos     : {modelos}")
    print(f"Semillas    : {semillas}")
    print(f"Test        : 80 test_pos (GT plata) + 40 test_neg (vacío)")
    print("-" * 74)

    pendientes, hechos = [], []
    for m in modelos:
        for s in semillas:
            if not os.path.exists(os.path.join(ruta_run(a.resultados, m, s),
                                               "resultado.json")):
                continue
            (hechos if (run_evaluado(a.resultados, m, s) and not a.sobrescribir)
             else pendientes).append((m, s))

    print(f"Plan: {len(pendientes)} por evaluar, {len(hechos)} ya evaluados")
    for m, s in pendientes:
        print(f"   {m:16s} semilla {s}")
    if a.verificar:
        verificar_checkpoints(modelos, semillas, a.resultados)
        return
    if a.dry_run:
        print("\n[dry-run] No se evalúa nada.")
        return

    errores = []
    for i, (m, s) in enumerate(pendientes, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(pendientes)}] {m} · semilla {s}", flush=True)
        try:
            r = evaluar_run(m, s, a.resultados, device, a.splits_dir,
                            a.batch, a.num_workers)
            if r.get("estado") != "completado":
                print(f"   omitido: {r.get('estado')}")
                errores.append((m, s, r.get("estado")))
                continue
            r["segundos_eval"] = round(time.time() - t0, 1)
            rjson = os.path.join(ruta_run(a.resultados, m, s), "resultado.json")
            original = leer_json(rjson)
            original.update(r)          # conserva historial y config
            escribir_json(rjson, original)
            print(f"   Dice_test={r['dice_test']:.4f}  "
                  f"IoU={r['iou_test']:.4f}  "
                  f"HD95={r['hd95_test']}  "
                  f"AUC-score listo ({r['n_test_pos']}+{r['n_test_neg']} imgs)")
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}")
            errores.append((m, s, f"{type(e).__name__}: {e}"))

    filas = escribir_resumen_test(a.resultados, modelos, semillas)
    print("\n" + "=" * 74)
    print(f"resumen_test.csv: {len(filas)} runs con metricas de test")
    if errores:
        print(f"ERRORES/OMITIDOS: {len(errores)}")
        for e in errores:
            print(f"   {e}")
    print("=" * 74)

    # --------------------------------------------------- aviso de duplicados
    # Si dos modelos tienen el MISMO sha256 de checkpoint o el MISMO CSV de
    # test, no son dos modelos distintos. Ocurrió con attention_unet por un
    # parámetro ignorado de smp; se avisa en voz alta para no publicar un
    # ranking con filas duplicadas.
    huellas = {}
    for m in modelos:
        for s in semillas:
            rcsv = os.path.join(ruta_run(a.resultados, m, s), "por_imagen_test.csv")
            if os.path.exists(rcsv):
                huellas.setdefault(sha256(rcsv), []).append(f"{m}/s{s}")
    dup = {h: v for h, v in huellas.items() if len(v) > 1}
    if dup:
        print("\n⚠ ATENCIÓN: predicciones idénticas entre modelos distintos:")
        for h, v in dup.items():
            print(f"   {', '.join(v)}  (sha256 {h[:12]}…)")
        print("   Un modelo duplicado NO debe entrar al ranking sin corregir.")


if __name__ == "__main__":
    main()
