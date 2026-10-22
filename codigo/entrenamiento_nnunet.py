"""
entrenamiento_nnunet.py — nnU-Net v2 como pipeline propio del benchmark (E2).

nnU-Net no es una red que se construya en `modelos_tb`: es un pipeline
completo (planificación automática de red/parches/normalización + 5-fold CV).
El benchmark lo integra con estas EQUIVALENCIAS DECLARADAS (ver
docs/NNUNET_PIPELINE.md):

  * Las 5 ejecuciones del protocolo (5 semillas) <-> los 5 folds internos de
    nnU-Net: fold 0->semilla 42, 1->123, 2->777, 3->2024, 4->31415.
  * Presupuesto acotado: trainer de 50 épocas (nnUNetTrainer_50epochs si el
    paquete lo trae; si no, se genera automáticamente un trainer propio con el
    mismo nombre).
  * Datos: mismo GT plata congelado que el resto. Pool train+val (194 npz) en
    imagesTr/labelsTr; test_pos (80) evaluado por fold contra
    ground_truth_test/.
  * Salida: `resultados/nnunet/seed_<semilla>/resultado.json` con el MISMO
    esquema del driver (estado/dice_val/...) + `por_imagen.csv` del test ⇒
    cuenta como run completado y `recomponer_resumen` lo incluye.

Uso (Colab, GPU):
    python codigo/entrenamiento_nnunet.py --resultados <RESULTADOS>

Reanudable: cada fold con resultado.json ya completado se omite.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driver_entrenamiento import (guardar_resultado, recomponer_resumen,  # noqa: E402
                                  run_completado, entorno_run)            # noqa: E402
from modelos_tb import ORDEN_BENCHMARK, ORDEN_BASELINES, SEMILLAS          # noqa: E402
from eda_lib_tb import encontrar_root, cargar_splits, ruta_local           # noqa: E402
from metricas_tb import metricas_imagen                                    # noqa: E402

DATASET_ID = "999"
NOMBRE_DS = f"Dataset{DATASET_ID}_TB"


# --------------------------------------------------------------------------- #
# Entorno y utilidades
# --------------------------------------------------------------------------- #
def preparar_entorno(raiz: Path, res_dir: Path):
    """Variables de entorno de nnU-Net v2: raw/preprocessed en disco local
    (rápido), resultados en el directorio de resultados del benchmark."""
    base = raiz / "nnunet_fs"
    rutas = {"nnUNet_raw": base / "raw",
             "nnUNet_preprocessed": base / "preprocessed",
             "nnUNet_results": res_dir,
             # alias: versiones tempranas de nnunetv2 leían este nombre
             "nnUNetv2_results": res_dir}
    for k, v in rutas.items():
        v.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(v)
    return rutas


def ejecutar(cmd, cwd=None):
    print("»", " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"comando falló ({proc.returncode}): {cmd[0]}")
    return time.time() - t0


def nnunet_cmd(modulo: str, *args):
    """CLI de nnU-Net v2 sin depender de que los console_scripts estén en
    PATH (en Colab/%pip no siempre pasan). `modulo` es el nombre del entry
    point; se resuelve a la ruta real del paquete nnunetv2 (>=2.6) vía
    python -m, con reserva al nombre del script si cambia la estructura."""
    rutas = {
        "plan_and_preprocess": (
            "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "nnUNetv2_plan_and_preprocess"),
        "run_training": ("nnunetv2.run.run_training", "nnUNetv2_train"),
        "predict_from_raw_data": (
            "nnunetv2.inference.predict_from_raw_data", "nnUNetv2_predict"),
    }
    mod, script = rutas[modulo]
    try:
        import importlib.util
        if importlib.util.find_spec(mod) is not None:
            return [sys.executable, "-m", mod, *args]
    except Exception:
        pass
    return [script, *args]


def asegurar_trainer(nombre: str, epocas: int) -> str:
    """Garantiza que `nombre` sea resoluble por nnU-Net (usa el mismo buscador
    canónico que el propio CLI). Si el paquete no lo trae, lo crea como
    subclase genérica de nnUNetTrainer con num_epochs=epocas (misma técnica
    que los nnUNetTrainer_Xepochs incluidos)."""
    from nnunetv2.utilities.find_objects import (
        recursive_find_trainer_class_by_name)
    try:
        cls = recursive_find_trainer_class_by_name(nombre)
    except Exception:
        cls = None            # versiones nuevas lanzan si no lo encuentra
    if cls is not None:
        print(f"trainer disponible: {nombre} ({cls.__module__})")
        return nombre

    import nnunetv2.training.nnUNetTrainer as pkg
    ruta = Path(pkg.__file__).parent / f"{nombre}.py"
    # La firma DEBE replicar la del trainer base: nnUNetTrainer.__init__
    # inspecciona self.__init__ y lee esos nombres de locals() (nnunetv2>=2.6).
    ruta.write_text(
        "import torch\n\n"
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer "
        "import nnUNetTrainer\n\n\n"
        f"class {nombre}(nnUNetTrainer):\n"
        "    def __init__(self, plans, configuration, fold, dataset_json,\n"
        "                 device=torch.device('cuda')):\n"
        "        super().__init__(plans, configuration, fold, dataset_json,\n"
        "                         device)\n"
        f"        self.num_epochs = {int(epocas)}\n",
        encoding="utf-8")
    importlib.invalidate_caches()
    try:
        ok = recursive_find_trainer_class_by_name(nombre) is not None
    except Exception:
        ok = False
    assert ok, f"no pude crear el trainer {nombre} en {ruta}"
    print(f"trainer propio creado: {nombre} ({epocas} épocas) -> {ruta}")
    return nombre


# --------------------------------------------------------------------------- #
# Conversión del dataset al formato nnU-Net
# --------------------------------------------------------------------------- #
def convertir_dataset(raiz: Path):
    """npz (train+val) + PNG test -> nnUNet_raw/Dataset999_TB (nii.gz).
    Idempotente: si el dataset ya existe, no reescribe nada."""
    import nibabel as nib

    env_raw = Path(os.environ["nnUNet_raw"]) / NOMBRE_DS
    imgs_tr, labs_tr = env_raw / "imagesTr", env_raw / "labelsTr"
    imgs_te = env_raw / "imagesTest"
    ds_json = env_raw / "dataset.json"

    if ds_json.exists():
        print(f"dataset nnU-Net ya convertido: {env_raw}")
        return env_raw

    for d in (imgs_tr, labs_tr, imgs_te):
        d.mkdir(parents=True, exist_ok=True)

    splits = cargar_splits(raiz)
    n = 0
    for split in ("train", "val"):
        for r in splits[split].ruta:
            z = np.load(ruta_local(r, raiz))
            _id = str(z["fname"]).replace(".png", "")
            img = z["image"].astype(np.float32)
            msk = (z["mask"] > 0).astype(np.uint8)
            nib.save(nib.Nifti1Image(img, np.eye(4)), str(imgs_tr / f"{_id}_0000.nii.gz"))
            nib.save(nib.Nifti1Image(msk, np.eye(4)), str(labs_tr / f"{_id}.nii.gz"))
            n += 1

    for r in splits["test_pos"].ruta:
        from PIL import Image
        _id = Path(r.replace("\\", "/")).stem
        g = np.asarray(Image.open(ruta_local(r, raiz)).convert("L"), dtype=np.float32)
        nib.save(nib.Nifti1Image(g, np.eye(4)), str(imgs_te / f"{_id}_0000.nii.gz"))

    ds_json.write_text(json.dumps({
        "channel_names": {"0": "Xray"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n,
        "file_ending": ".nii.gz",
    }, indent=2), encoding="utf-8")
    print(f"convertidos {n} npz (train+val) y {len(splits['test_pos'])} test -> {env_raw}")
    return env_raw


# --------------------------------------------------------------------------- #
# Dice de validación interna del fold (resumen de nnU-Net o cálculo directo)
# --------------------------------------------------------------------------- #
def metricas_val_fold(exp_dir: Path, fold: int):
    """Validación interna del fold (sobre el pool de 194).

    Devuelve (dice_medio, filas_por_imagen) con la MISMA convención que el
    driver: por_imagen.csv son métricas de val. 1) intenta
    validation/summary.json (solo el escalar); 2) calcula pred vs gt_niftis
    con metricas_imagen (escalar + filas)."""
    val = exp_dir / f"fold_{fold}" / "validation"
    escalar = None
    resumen = val / "summary.json"
    if resumen.exists():
        try:
            with open(resumen, encoding="utf-8") as f:
                escalar = float(json.load(f)["results"]["mean"]["Dice"])
        except (KeyError, TypeError, ValueError):
            escalar = None

    gts = sorted((val / "gt_niftis").glob("*.nii.gz"))
    preds = {p.name[len("pred_"):]: p for p in val.glob("pred_*.nii.gz")}
    if not gts:
        return escalar, []
    import nibabel as nib
    filas = []
    for g in gts:
        p = preds.get(g.name)
        if p is None:
            continue
        a = (nib.load(str(g)).get_fdata() > 0.5).astype(np.uint8)
        b = (nib.load(str(p)).get_fdata() > 0.5).astype(np.uint8)
        m = metricas_imagen(b, a)
        m["id"] = g.name.replace(".nii.gz", "")
        m["tb_type"] = "na"
        filas.append(m)
    if not filas:
        return escalar, []
    if escalar is None:
        escalar = float(np.mean([x["dice"] for x in filas]))
    return escalar, filas


# --------------------------------------------------------------------------- #
# Evaluación del test_pos con una predicción por fold
# --------------------------------------------------------------------------- #
def evaluar_test(pred_dir: Path, raiz: Path, carpeta_run: Path):
    """Predicciones del fold contra el GT plata; escribe por_imagen_TEST.csv
    (la convención por_imagen.csv queda para val, igual que el driver) y
    devuelve el Dice medio de test."""
    import nibabel as nib

    splits = cargar_splits(raiz)
    gt_dir = raiz / "datos" / "ground_truth_test"
    filas = []
    for r in splits["test_pos"].ruta:
        _id = Path(r.replace("\\", "/")).stem
        pred_nii = pred_dir / f"{_id}_0000.nii.gz"
        if not pred_nii.exists():
            continue
        pred = (nib.load(str(pred_nii)).get_fdata() > 0.5).astype(np.uint8)
        from PIL import Image
        gt = (np.asarray(Image.open(gt_dir / f"{_id}.png")) > 127).astype(np.uint8)
        m = metricas_imagen(pred, gt)
        m["id"] = _id
        m["tb_type"] = "na"
        filas.append(m)
    if not filas:
        raise RuntimeError(f"sin predicciones de test en {pred_dir}")
    carpeta_run.mkdir(parents=True, exist_ok=True)
    with open(carpeta_run / "por_imagen_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    return float(np.mean([x["dice"] for x in filas]))


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main():
    raiz = encontrar_root()
    ap = argparse.ArgumentParser(description="nnU-Net v2 (pipeline propio del benchmark)")
    ap.add_argument("--resultados", default=str(raiz / "resultados"),
                    help="Directorio de resultados del benchmark (Drive en Colab)")
    ap.add_argument("--splits-dir", default=str(raiz / "datos" / "splits"),
                    help="Solo informativo: los splits se leen siempre desde "
                         "<raíz>/datos/splits (§R1.8, la membresía la define "
                         "el CSV)")
    ap.add_argument("--red", default="2d", choices=("2d", "3d_fullres"))
    ap.add_argument("--trainer", default="nnUNetTrainer_50epochs")
    ap.add_argument("--epocas", type=int, default=50,
                    help="Épocas si hay que crear el trainer propio")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="Folds a entrenar (default: los pendientes)")
    ap.add_argument("--solo-preparar", action="store_true",
                    help="Conversión + planificación y salir (smoke test)")
    a = ap.parse_args()

    res_dir = Path(a.resultados).resolve()
    env = preparar_entorno(raiz, res_dir)
    trainer = asegurar_trainer(a.trainer, a.epocas)

    # 1) dataset + planificación (una vez)
    convertir_dataset(raiz)
    plans = env["nnUNet_preprocessed"] / NOMBRE_DS / "nnUNetPlans.json"
    if not plans.exists():
        ejecutar(nnunet_cmd("plan_and_preprocess", "-d", DATASET_ID,
                            "--verify_dataset_integrity"))
    if a.solo_preparar:
        print("solo-preparar: dataset listo y planificado.")
        return

    # 2) folds pendientes
    folds = a.folds if a.folds is not None else list(range(len(SEMILLAS)))
    exp_dir = res_dir / NOMBRE_DS / f"{trainer}__nnUNetPlans__{a.red}"
    for fold in folds:
        semilla = SEMILLAS[fold]
        if run_completado(res_dir, "nnunet", semilla):
            print(f"fold {fold} (semilla {semilla}): ya completado, se omite")
            continue

        t0 = time.time()
        print(f"\n=== fold {fold} <-> semilla {semilla} · {a.red} · {trainer} ===")
        ejecutar(nnunet_cmd("run_training", DATASET_ID, a.red, str(fold),
                            "-tr", trainer))

        dv, filas_val = metricas_val_fold(exp_dir, fold)

        carpeta_run = res_dir / "nnunet" / f"seed_{semilla}"
        # convención del driver: por_imagen.csv = val (aquí, fold interno)
        if filas_val:
            carpeta_run.mkdir(parents=True, exist_ok=True)
            with open(carpeta_run / "por_imagen.csv", "w", newline="",
                      encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(filas_val[0].keys()))
                w.writeheader()
                w.writerows(filas_val)
        pred_dir = carpeta_run / "pred_test"
        n_test = len(cargar_splits(raiz)["test_pos"])
        n_pred = len(list(pred_dir.glob("*_0000.nii.gz")))
        if n_pred < n_test:
            ejecutar(nnunet_cmd("predict_from_raw_data", "-d", DATASET_ID,
                                "-c", a.red,
                                "-i", env["nnUNet_raw"] / NOMBRE_DS / "imagesTest",
                                "-o", pred_dir, "-f", str(fold), "-tr", trainer,
                                "-chk", "checkpoint_final.pth"))
        dice_test = evaluar_test(pred_dir, raiz, carpeta_run)

        datos = {"estado": "completado", "modelo": "nnunet", "semilla": semilla,
                 "dice_val": None if dv is None else round(dv, 5),
                 "dice_test": round(dice_test, 5),
                 "fold": fold, "red": a.red, "trainer": trainer,
                 "epoch_best": None, "epochs_entrenadas": a.epocas,
                 "segundos": round(time.time() - t0, 1),
                 "device": "cuda" if _hay_cuda() else "cpu",
                 "entorno": entorno_run(),
                 "config": {"fold": fold, "epocas": a.epocas, "dataset": NOMBRE_DS,
                            "gt": "plata SAM2 congelado (D1-R1)",
                            "pool_train_val": 194, "test_pos": 80}}
        guardar_resultado(res_dir, "nnunet", semilla, datos)
        print(f"fold {fold} guardado: dice_val={datos['dice_val']} "
              f"dice_test={datos['dice_test']}")

    # 3) resumen unificado (incluye los otros modelos ya completados)
    recomponer_resumen(res_dir, ORDEN_BENCHMARK + ORDEN_BASELINES, SEMILLAS)
    print("\nresumen_runs.csv regenerado con todos los runs (incluye nnunet).")


def _hay_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
