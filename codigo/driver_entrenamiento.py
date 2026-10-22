"""
================================================================================
DRIVER DE ENTRENAMIENTO REANUDABLE - BENCHMARK ENTREGABLE 2
================================================================================

PROPÓSITO:
    Orquesta las 50 corridas (10 modelos x 5 semillas) de forma REANUDABLE,
    pensado para los cortes de sesión de Colab Pro (~90 min inactividad,
    ~12 h máximas por sesión).

DISEÑO ANTI-DESCONEXIÓN:
    1. Antes de entrenar, lista los runs ya completados en RESULT_DIR.
    2. Si el resultado de un run existe y es válido, lo SALTA.
    3. Al terminar cada run escribe el resultado en disco inmediatamente.
    => Una caída de sesión pierde como mucho la corrida en curso.

    Además: --max-minutos corta limpio tras N minutos y guarda estado,
    para encajar en ventanas de sesión cortas sin perder nada.

SALIDA POR RUN (en RESULT_DIR/<modelo>/seed_<n>/):
    - resultado.json      metricas agregadas del run
    - por_imagen.csv      metricas por imagen (para los tests por pares)
    - mejor_modelo.pt     checkpoint de la mejor época (val Dice)
    - run.log             log completo

RESULTADO AGREGADO (en RESULT_DIR/resumen_runs.csv):
    Una fila por run, para alimentar directamente las Tablas A/B/C.

USO:
    python driver_entrenamiento.py                       # todo lo que falte
    python driver_entrenamiento.py --modelos unet unet_pp
    python driver_entrenamiento.py --semillas 42
    python driver_entrenamiento.py --max-minutos 300     # corta a los 5 h
    python driver_entrenamiento.py --dry-run             # solo planifica
================================================================================
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- locales ---------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelos_tb import (MODELOS, ORDEN_BENCHMARK, ORDEN_BASELINES, SEMILLAS,
                        crear_modelo, resumen_entorno)      # noqa: E402
from dataset_tb import DatasetTB, SPLITS_DIR                # noqa: E402
import metricas_tb as M                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================================

CONFIG = {
    "epochs": 150,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "early_stopping_patience": 25,
    "criterio_seleccion": "dice_val",       # qué época se guarda
    "num_workers": 2,
    "umbral": 0.5,
    # ratio de la pérdida: se declara en el preregistro
    "peso_bce": 0.5,
    "peso_dice": 0.5,
}


# ============================================================================
# ENTORNO POR RUN (reproducibilidad: cada resultado.json lleva sus versiones)
# ============================================================================

def entorno_run():
    """Versiones exactas del runtime de ESTE run; se guardan en resultado.json.

    Permite verificar qué versiones produjeron cada métrica (registro de
    experimentos del Entregable 2). Devuelve None si un paquete no existe,
    para no tumbar el run por un problema de metadatos.
    """
    from importlib import metadata

    def _v(paquete):
        try:
            return metadata.version(paquete)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_disponible": bool(torch.cuda.is_available()),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "numpy": _v("numpy"),
        "segmentation_models_pytorch": _v("segmentation-models-pytorch"),
        "monai": _v("monai"),
        "nnunetv2": _v("nnunetv2"),
        "timm": _v("timm"),
    }


# ============================================================================
# PÉRDIDA (idéntica para todos los modelos - requisito del Entregable)
# ============================================================================

class DiceBCELoss(nn.Module):
    """
    BCEWithLogits + Soft Dice, combinados.

    Se usa la MISMA pérdida para los 10 modelos: si cada modelo tuviera su
    propia función de objetivo, la comparación dejaría de ser justa.
    """

    def __init__(self, peso_bce=0.5, peso_dice=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.peso_bce = peso_bce
        self.peso_dice = peso_dice
        self.smooth = smooth

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + self.smooth
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth
        dice = 1.0 - (num / den).mean()
        return self.peso_bce * bce + self.peso_dice * dice


# ============================================================================
# UTILIDADES DE PERSISTENCIA (lo que sobrevive a una desconexión)
# ============================================================================

def ruta_run(res_dir, modelo, semilla):
    return os.path.join(res_dir, modelo, f"seed_{semilla}")


def run_completado(res_dir, modelo, semilla):
    """Un run cuenta como hecho si su resultado.json existe y es válido."""
    ruta = os.path.join(ruta_run(res_dir, modelo, semilla), "resultado.json")
    if not os.path.exists(ruta):
        return False
    try:
        with open(ruta, encoding="utf-8") as f:
            r = json.load(f)
        return r.get("estado") == "completado" and "dice_val" in r
    except Exception:
        return False


def guardar_resultado(d, modelo, semilla, datos):
    rd = ruta_run(d, modelo, semilla)
    os.makedirs(rd, exist_ok=True)
    datos["guardado_en"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(rd, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    return rd


def anadir_resumen(d, fila):
    """Acumula una fila por run en resumen_runs.csv (append, no overwrite)."""
    ruta = os.path.join(d, "resumen_runs.csv")
    campos = list(fila.keys())
    existe = os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if not existe:
            w.writeheader()
        w.writerow(fila)


def recomponer_resumen(res_dir, modelos, semillas):
    """
    Regenera resumen_runs.csv a partir de los resultado.json individuales.

    Útil si el append se corrompió: el resumen se deriva siempre de los
    archivos de run, nunca al revés.
    """
    filas = []
    for m in modelos:
        for s in semillas:
            ruta = os.path.join(ruta_run(res_dir, m, s), "resultado.json")
            if not os.path.exists(ruta):
                continue
            with open(ruta, encoding="utf-8") as f:
                r = json.load(f)
            filas.append({
                "modelo": m, "semilla": s, "estado": r.get("estado"),
                "dice_val": r.get("dice_val"),
                "dice_test": r.get("dice_test"),
                "ood_frac_media": (r.get("ood") or {}).get("frac_media"),
                "epoch_best": r.get("epoch_best"),
                "epochs_entrenadas": r.get("epochs_entrenadas"),
                "segundos": r.get("segundos"), "device": r.get("device"),
                "mejor_modelo": os.path.join(ruta_run(res_dir, m, s),
                                             "mejor_modelo.pt"),
            })
    ruta = os.path.join(res_dir, "resumen_runs.csv")
    if filas:
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
    return filas


# ============================================================================
# UNA CORRIDA
# ============================================================================

def entrenar_run(modelo, semilla, res_dir, cfg, device, limite_segundos=None,
                 splits_dir=SPLITS_DIR):
    """Entrena UN modelo con UNA semilla. Devuelve el dict de resultado."""
    t0 = time.time()
    random.seed(semilla)
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False       # reproducibilidad > velocidad

    rd = ruta_run(res_dir, modelo, semilla)
    os.makedirs(rd, exist_ok=True)

    # ------------------------------------------------------------- datasets
    ds_tr = DatasetTB("train", splits_dir=splits_dir, augmentar=True,
                      verbose=False)
    ds_va = DatasetTB("val", splits_dir=splits_dir, augmentar=False,
                      verbose=False)
    g = torch.Generator()
    g.manual_seed(semilla)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                       num_workers=cfg["num_workers"], generator=g,
                       drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                       num_workers=cfg["num_workers"])

    # --------------------------------------------------------------- modelo
    try:
        net, meta = crear_modelo(modelo, in_channels=1, out_channels=1,
                                 verbose=False)
    except Exception as e:
        # Falta de dependencia NO debe tumbar el driver: se registra y sigue
        return {"estado": "no_disponible", "error": str(e).splitlines()[0],
                "modelo": modelo, "semilla": semilla}
    net = net.to(device)

    criterio = DiceBCELoss(cfg["peso_bce"], cfg["peso_dice"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    mejor_dice, mejor_epoch = -1.0, -1
    paciencia = 0
    historial = []

    net.train()
    for ep in range(1, cfg["epochs"] + 1):
        # ---- límite de tiempo: corte limpio para encajar en sesión de Colab
        if limite_segundos and (time.time() - t0) > limite_segundos:
            return {"estado": "interrumpido_por_tiempo", "modelo": modelo,
                    "semilla": semilla, "epochs_entrenadas": ep - 1,
                    "dice_val": mejor_dice, "epoch_best": mejor_epoch,
                    "segundos": round(time.time() - t0, 1), "device": str(device),
                    "entorno": entorno_run(), "historial": historial}

        ep_loss = 0.0
        for xb, yb, _ in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = criterio(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * xb.size(0)
        sched.step()
        ep_loss /= max(len(ds_tr), 1)

        # ---------------------------------------------------- validación
        net.eval()
        dices = []
        with torch.no_grad():
            for xb, yb, _ in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = net(xb)
                prob = torch.sigmoid(logits)
                for p_, y_ in zip(prob, yb):
                    dices.append(M.dice_pytorch(p_[0] > cfg["umbral"],
                                                y_[0] > 0.5))
        net.train()
        dice_val = float(np.mean(dices)) if dices else 0.0
        historial.append({"epoch": ep, "loss": round(ep_loss, 5),
                          "dice_val": round(dice_val, 5)})

        # -------------------------------------------------- checkpointing
        if dice_val > mejor_dice:
            mejor_dice, mejor_epoch = dice_val, ep
            paciencia = 0
            torch.save({"state_dict": net.state_dict(), "modelo": modelo,
                        "semilla": semilla, "epoch": ep,
                        "dice_val": dice_val, "config": cfg},
                       os.path.join(rd, "mejor_modelo.pt"))
        else:
            paciencia += 1

        if ep % 5 == 0 or paciencia == 0:
            print(f"    [{modelo} s{semilla}] época {ep:3d}  "
                  f"loss={ep_loss:.4f}  dice_val={dice_val:.4f}  "
                  f"(mejor {mejor_dice:.4f} @ {mejor_epoch})", flush=True)

        if paciencia >= cfg["early_stopping_patience"]:
            print(f"    [{modelo} s{semilla}] early stop en época {ep}", flush=True)
            break

    # ----------------------------------------------- métricas en validación
    net.eval()
    filas = []
    with torch.no_grad():
        for xb, yb, infos in dl_va:
            xb, yb = xb.to(device), yb.to(device)
            prob = torch.sigmoid(net(xb))
            for k in range(xb.size(0)):
                p = (prob[k, 0] > cfg["umbral"]).cpu().numpy().astype(np.uint8)
                gt = (yb[k, 0] > 0.5).cpu().numpy().astype(np.uint8)
                m = M.metricas_imagen(p, gt)
                m["id"] = infos["id"][k]
                m["tb_type"] = infos["tb_type"][k]
                filas.append(m)
    ruta_csv = os.path.join(rd, "por_imagen.csv")
    if filas:
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)

    return {"estado": "completado", "modelo": modelo, "semilla": semilla,
            "dice_val": round(mejor_dice, 5), "epoch_best": mejor_epoch,
            "epochs_entrenadas": len(historial),
            "segundos": round(time.time() - t0, 1), "device": str(device),
            "entorno": entorno_run(),
            "n_val": len(ds_va), "historial": historial,
            "config": cfg}


# ============================================================================
# DRIVER
# ============================================================================

def planificar(modelos, semillas, res_dir):
    """Devuelve (pendientes, hechos) sin tocar nada."""
    pendientes, hechos = [], []
    for m in modelos:
        for s in semillas:
            (hechos if run_completado(res_dir, m, s) else pendientes).append((m, s))
    return pendientes, hechos


def main():
    ap = argparse.ArgumentParser(
        description="Driver reanudable del benchmark (Entregable 2)")
    ap.add_argument("--modelos", nargs="*", default=None,
                    help="Claves a entrenar (por defecto: todas)")
    ap.add_argument("--semillas", nargs="*", type=int, default=None,
                    help=f"Semillas (por defecto: {SEMILLAS})")
    ap.add_argument("--resultados", default=os.path.join(ROOT, "resultados"),
                    help="Directorio de resultados (montar en Drive en Colab)")
    ap.add_argument("--splits-dir", default=SPLITS_DIR,
                    help="Directorio de los CSV de split (en Colab apuntar al "
                         "copia local remapeada, ver setup_colab.ipynb)")
    ap.add_argument("--max-minutos", type=float, default=None,
                    help="Corta limpio tras N minutos (para sesiones de Colab)")
    ap.add_argument("--cpu", action="store_true",
                    help="Permite entrenar sin GPU (solo smoke tests: "
                         "extremadamente lento para el benchmark completo)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo muestra el plan, no entrena")
    ap.add_argument("--recomponer-resumen", action="store_true",
                    help="Regenera resumen_runs.csv y sale")
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--batch", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--lr", type=float, default=CONFIG["lr"],
                    help="Override de learning rate (para la grilla de "
                         "unet_fuerte declarada en la Sección 4.1)")
    ap.add_argument("--weight-decay", type=float,
                    default=CONFIG["weight_decay"],
                    help="Override de weight decay (ídem)")
    a = ap.parse_args()

    # Consolas Windows (cp1252) no soportan ✓/⚠/✗: forzar UTF-8 en la salida
    # para que el informe no aborte a mitad de camino.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    modelos = a.modelos or (ORDEN_BENCHMARK + ORDEN_BASELINES)
    semillas = a.semillas or SEMILLAS
    # los baselines sin red no se entrenan
    modelos = [m for m in modelos if MODELOS.get(m, {}).get("entrenable", False)]
    # nnU-Net es un pipeline propio (5 folds internos = las 5 ejecuciones):
    # se entrena con su script y escribe el MISMO formato de resultado.json.
    if "nnunet" in modelos:
        modelos = [m for m in modelos if m != "nnunet"]
        print("NOTA: nnunet queda fuera de este loop (pipeline propio).\n"
              "      Ejecutar por separado:\n"
              "        python codigo/entrenamiento_nnunet.py "
              "--resultados <RESULTADOS>\n"
              "      (escribe resultados/nnunet/seed_<semilla>/resultado.json "
              "por fold y\n      regenera resumen_runs.csv; reanudable por "
              "fold).")

    cfg = dict(CONFIG)
    cfg["epochs"] = a.epochs
    cfg["batch_size"] = a.batch
    cfg["lr"] = a.lr
    cfg["weight_decay"] = a.weight_decay
    if cfg["batch_size"] < 2:
        print("⚠ batch<2: DeepLabV3+ falla (BN sobre 1x1 en image pooling) "
              "y BatchNorm de otros modelos es inestable. Usa --batch >= 2.")
        if not a.dry_run:
            return

    res_dir = a.resultados
    os.makedirs(res_dir, exist_ok=True)

    if a.recomponer_resumen:
        filas = recomponer_resumen(res_dir, modelos, semillas)
        print(f"resumen_runs.csv regenerado con {len(filas)} filas")
        return

    # ------------------------------------------------------- diagnóstico
    print("=" * 74)
    print("DRIVER DE ENTRENAMIENTO - ENTREGABLE 2")
    print("=" * 74)
    print(f"Hora inicio : {datetime.now().isoformat(timespec='seconds')}")
    if torch.cuda.is_available():
        _dev = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        _dev = "CPU (¡sin GPU!)"
    print(f"Device       : {_dev}")
    if not torch.cuda.is_available():
        print("  ⚠  No hay GPU. El benchmark completo en CPU no es viable.")
        print("     En Colab: Runtime -> Change runtime type -> T4 o L4.")
    print(f"Resultados   : {res_dir}")
    print(f"Config       : epochs={cfg['epochs']} batch={cfg['batch_size']} "
          f"lr={cfg['lr']} patience={cfg['early_stopping_patience']}")
    print(f"Modelos      : {len(modelos)}")
    print(f"Semillas     : {semillas}")

    diag, _ = resumen_entorno()

    # ----------------------------------------------------------- plan
    pendientes, hechos = planificar(modelos, semillas, res_dir)
    print("-" * 74)
    print(f"Plan: {len(pendientes)} corridas pendientes, "
          f"{len(hechos)} ya completadas (se saltan)")
    for m, s in pendientes:
        ok = diag.get(m, (False, ""))[0]
        marca = " " if ok else "⚠"
        print(f"   {marca} {m:16s} semilla {s}")
    if a.dry_run:
        print("\n[dry-run] No se entrena nada.")
        return

    if not pendientes:
        print("\nNada pendiente. Todo el benchmark ya está corrido.")
        return

    if not torch.cuda.is_available() and not a.cpu:
        print("\n⚠  Abortado: sin GPU. Instala/configura CUDA o usa Colab.")
        print("   (para un smoke test en CPU usa --cpu --epochs 1)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limite = a.max_minutos * 60 if a.max_minutos else None
    if limite:
        print(f"\nLímite de sesión: {a.max_minutos} min "
              f"(corte limpio, reanudable)")

    # ------------------------------------------------------ ejecución
    t_total = time.time()
    completadas, fallidas, saltadas = 0, [], 0
    for i, (modelo, semilla) in enumerate(pendientes, 1):
        if limite and (time.time() - t_total) > limite:
            print(f"\n⏱  Límite de sesión alcanzado tras {completadas} corridas.")
            print("   Estado guardado. Reejecuta el mismo comando para seguir.")
            break

        print(f"\n[{i}/{len(pendientes)}] {modelo} · semilla {semilla}", flush=True)
        if not diag.get(modelo, (False, ""))[0]:
            print("    ⚠ dependencia ausente -> se registra y se omite")
            guardar_resultado(res_dir, modelo, semilla,
                              {"estado": "no_disponible",
                               "modelo": modelo, "semilla": semilla,
                               "error": diag.get(modelo, (None, ""))[1]})
            fallidas.append((modelo, semilla))
            continue

        try:
            r = entrenar_run(modelo, semilla, res_dir, cfg, device,
                             limite_segundos=limite,
                             splits_dir=a.splits_dir)
        except KeyboardInterrupt:
            print("\n Interrumpido por el usuario. Estado guardado donde haya.")
            raise
        except Exception as e:
            r = {"estado": "error", "modelo": modelo, "semilla": semilla,
                 "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc()[-2000:]}
            print(f"    ✗ ERROR: {r['error']}")

        guardar_resultado(res_dir, modelo, semilla, r)
        anadir_resumen(res_dir, {"modelo": modelo, "semilla": semilla,
                                 "estado": r.get("estado"),
                                 "dice_val": r.get("dice_val"),
                                 "epoch_best": r.get("epoch_best"),
                                 "epochs_entrenadas": r.get("epochs_entrenadas"),
                                 "segundos": r.get("segundos"),
                                 "device": r.get("device")})
        if r.get("estado") == "completado":
            completadas += 1
            print(f"    ✓ dice_val={r['dice_val']:.4f} "
                  f"({r['segundos']} s)")
        else:
            fallidas.append((modelo, semilla))
            print(f"    ⚠ estado: {r.get('estado')}")

    # ---------------------------------------------------------- resumen
    print("\n" + "=" * 74)
    print("RESUMEN DE LA SESIÓN")
    print("=" * 74)
    print(f"  Completadas : {completadas}")
    print(f"  Fallidas    : {len(fallidas)}")
    for m, s in fallidas:
        print(f"      - {m} (semilla {s})")
    print(f"  Duración    : {round((time.time() - t_total) / 60, 1)} min")
    print(f"  Resultados  : {res_dir}")
    print(f"  Reanudar    : python driver_entrenamiento.py "
          f"--resultados \"{res_dir}\"")
    if fallidas:
        print("\n  Las corridas fallidas se reintentan al volver a ejecutar "
              "el mismo comando.")


if __name__ == "__main__":
    main()
