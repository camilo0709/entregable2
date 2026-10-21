"""
================================================================================
DATASET - CARGA DESDE SPLITS SELLADOS (ENTREGABLE 2)
================================================================================

PROPÓSITO:
    Carga train/val desde los CSV sellados en datos/splits/. Separa con
    claridad QUÉ máscara se usa en cada split, para que la circularidad
    del GT quede imposible de ocultar en el log:

        train/val  -> 'mask'      (pseudomáscara SAM 2)      [etiqueta débil]
        test_pos   -> ground_truth_test/<id>.png              [D1-R1: GT plata]

    REVISIÓN D1-R1 (2026-10-20): el GT del test_pos pasó de "manual humana"
    a "plata" (pseudomáscara SAM 2 con el pipeline congelado
    codigo/generar_pseudo_gt.py). Es un silver standard DECLARADO — ver
    docs/DECISION_D1_GROUND_TRUTH.md §R1 — con validación humana de una
    submuestra como techo de desempeño.

    Si el GT del test no existe, el dataset lo dice en voz alta y se niega
    a fabricar uno implícito.

    NOTA DE RUTAS: los CSV sellados contienen rutas absolutas Windows
    (C:\\Users\\...). Los CSV NO se modifican (sus SHA-256 están en el
    manifest); el remapeo a rutas locales ocurre solo al cargar
    (_ruta_local).

PROTOCOLO:
    - Imagen: 1 canal, float32, normalizada por imagen (z-score)
    - Máscara: binaria {0,1}, float32
    - Sin augmentation en la carga: se aplica en el dataloader de forma
      determinista por semilla (ver entrenamiento.py)

USO:
    ds = DatasetTB("train", "datos/splits")
    img, msk, info = ds[0]
================================================================================
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Rutas relativas al raíz del proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(ROOT, "datos", "splits")
GT_DIR = os.path.join(ROOT, "datos", "ground_truth_test")
NPZ_DIR = os.path.join(ROOT, "datos", "npz_dataset")
TBX_TB = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K", "imgs", "tb")
TBX_HEALTH = os.path.join(ROOT, "datos", "kaggle_dataset", "TBX11K", "imgs", "health")


def _leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ruta_local(ruta):
    """
    Convierte las rutas absolutas Windows de los CSV sellados a rutas
    locales del proyecto. Los CSV jamás se editan (hash integrity vs
    split_manifest.json); el remapeo es SOLO de lectura.

    Si la ruta ya existe tal cual (p. ej. en la máquina Windows original
    o tras la remapación de setup_colab.ipynb en Colab), se devuelve
    sin cambios.
    """
    if not ruta or os.path.exists(ruta):
        return ruta
    if len(ruta) > 2 and ruta[1] == ":" :          # C:\... o D:/...
        norm = ruta.replace("\\", "/")
        i = norm.find("datos/")
        if i >= 0:
            candidato = os.path.join(ROOT, norm[i:])
            if os.path.exists(candidato):
                return candidato
            # también puede empezar por codigo/ o modelos/
            return candidato                        # error claro si falta
    return ruta


def _normalizar(img):
    """z-score por imagen. Establece un rango común entre radiografías."""
    img = img.astype(np.float32)
    mu, sd = float(img.mean()), float(img.std())
    if sd < 1e-6:
        sd = 1.0
    return (img - mu) / sd


def normalizar_imagen_np(img):
    """Igual que _normalizar pero accesible desde fuera (inferencia)."""
    return _normalizar(img)


# ============================================================================
# DATASETS
# ============================================================================

class DatasetTB(Dataset):
    """
    Dataset unificado para los tres splits del Entregable 2.

    split:
        "train"   - NPZ, máscara SAM 2, augmentation activa
        "val"     - NPZ, máscara SAM 2, SIN augmentation
        "test_pos"- PNG TBX11K, GT plata SAM 2 (ground_truth_test/, D1-R1)
        "test_neg"- PNG TBX11K sano, máscara vacía por construcción
    """

    SPLITS_DIR = SPLITS_DIR

    def __init__(self, split, splits_dir=None, augmentar=False,
                 transforms=None, esperar_gt=True, verbose=True):
        self.split = split
        self.splits_dir = splits_dir or SPLITS_DIR
        self.augmentar = augmentar
        self.transforms = transforms

        ruta_csv = os.path.join(self.splits_dir, f"{split}.csv")
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(
                f"No existe el split '{split}'.\n"
                f"  Esperado en: {ruta_csv}\n"
                f"  Genera los splits con: "
                f"python codigo/generar_split_entregable2.py"
            )
        self.filas = _leer_csv(ruta_csv)
        if not self.filas:
            raise ValueError(f"El split '{split}' está vacío: {ruta_csv}")

        # --- qué máscara usa cada split (requisito D1, revisado en R1) ----
        if split in ("train", "val"):
            self.fuente_gt = "pseudo_sam2"
            self._cargar_mask = self._mask_desde_npz
        elif split == "test_pos":
            # D1-R1: GT plata (silver standard declarado). Generado con el
            # pipeline congelado codigo/generar_pseudo_gt.py (SAM 2 con
            # prompt de caja). Ver docs/DECISION_D1_GROUND_TRUTH.md §R1.
            self.fuente_gt = "pseudo_sam2_d1r1"
            self._cargar_mask = self._mask_manual
            if esperar_gt and not os.path.isdir(GT_DIR):
                raise FileNotFoundError(
                    "Falta el GT del test_pos (plata, D1-R1).\n"
                    f"  Directorio esperado: {GT_DIR}\n"
                    "  Generalo con el pipeline congelado:\n"
                    "    python codigo/generar_pseudo_gt.py --splits test_pos\n"
                    "  (requiere checkpoint SAM 2: --descargar-ckpt; GPU opcional)\n"
                    "  o pasa esperar_gt=False solo para pruebas de carga."
                )
            if not os.path.isdir(GT_DIR):
                # esperar_gt=False: se permite probar la CARGA con una
                # máscara vacía marcada como ausente; NUNCA sirve para
                # evaluar (el umbral de evaluación es estricto por diseño).
                self.fuente_gt = "ausente_solo_carga"
                self._cargar_mask = self._mask_vacia
                if verbose:
                    print("  ⚠ GT del test ausente: test_pos cargará máscaras "
                          "vacías SOLO para prueba de carga (no evaluar)")
        elif split == "test_neg":
            self.fuente_gt = "vacia_por_construccion"
            self._cargar_mask = self._mask_vacia
        else:
            raise ValueError(f"Split desconocido: {split!r}")

        if verbose:
            print(f"[{split}] {len(self.filas)} imágenes | "
                  f"GT = {self.fuente_gt}")

    # ----------------------------------------------------------- máscaras
    def _mask_desde_npz(self, ruta_npz):
        with np.load(ruta_npz, allow_pickle=True) as z:
            return z["mask"].astype(np.float32)

    def _mask_vacia(self, _ruta):
        return np.zeros((512, 512), dtype=np.float32)

    def _mask_manual(self, fila):
        """Carga el GT del test desde ground_truth_test/<id>.png
        (D1-R1: GT plata SAM 2; con --anotador-b, la máscara humana
        de la submuestra de validación)."""
        iid = fila["id"]
        ruta = os.path.join(GT_DIR, f"{iid}.png")
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"GT del test ausente para {iid}: {ruta}\n"
                f"  ({len(self.filas)} imágenes en test_pos; "
                f"faltan las que no existan en disco)"
            )
        from PIL import Image
        m = np.array(Image.open(ruta).convert("L"))
        # el GT se guarda como {0,255}; se convierte a {0,1}
        return (m > 127).astype(np.float32)

    # ------------------------------------------------------------ imágenes
    def _imagen_desde_npz(self, ruta_npz):
        with np.load(ruta_npz, allow_pickle=True) as z:
            return z["image"].astype(np.float32)

    def _imagen_desde_png(self, ruta):
        from PIL import Image
        return np.array(Image.open(ruta).convert("L")).astype(np.float32)

    # --------------------------------------------------------------- get
    def __len__(self):
        return len(self.filas)

    def __getitem__(self, i):
        fila = self.filas[i]
        ruta = _ruta_local(fila["ruta"])

        if self.split in ("train", "val"):
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"NPZ ausente: {ruta}")
            img = self._imagen_desde_npz(ruta)
        else:
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"Imagen ausente: {ruta}")
            img = self._imagen_desde_png(ruta)

        msk = self._cargar_mask(ruta if self.split in ("train", "val") else fila)

        if img.shape != msk.shape:
            raise ValueError(
                f"Desalineación imagen/máscara en {fila['id']}: "
                f"{img.shape} vs {msk.shape}"
            )

        img = _normalizar(img)

        # augmentation determinista por semilla+índice (solo train)
        if self.augmentar:
            img, msk = self._augmentar(img, msk, i)

        info = {
            "id": fila["id"],
            "split": self.split,
            "gt_tipo": fila.get("gt_tipo", ""),
            "tb_type": fila.get("tb_type", ""),
            "n_boxes": int(fila.get("n_boxes", 0) or 0),
            "indice": i,
        }
        if "box_area_mean" in fila and fila["box_area_mean"]:
            info["box_area_mean"] = float(fila["box_area_mean"])

        return (torch.from_numpy(img[None, ...]),      # (1,H,W)
                torch.from_numpy(msk[None, ...]),      # (1,H,W)
                info)

    # ------------------------------------------------------- augmentation
    def _augmentar(self, img, msk, idx):
        """
        Augmentation ligera y determinista. Se aplica SOLO en train.

        Sin ella el modelo con 165 imágenes sobreajusta rápido; pero se
        mantiene conservadora para no inflar artificialmente el rendimiento.
        """
        rng = np.random.RandomState(1000 + idx)   # fijo por índice
        if rng.rand() < 0.5:
            img = img[:, ::-1].copy()              # flip horizontal
            msk = msk[:, ::-1].copy()
        if rng.rand() < 0.3:
            img = img[::-1, :].copy()              # flip vertical suave
            msk = msk[::-1, :].copy()
        if rng.rand() < 0.3:
            brillo = rng.normal(1.0, 0.05)         # contraste
            img = np.clip(img * brillo, -4, 4)
        if rng.rand() < 0.2:
            img = img + rng.normal(0, 0.03, img.shape).astype(np.float32)
        return img.astype(np.float32), msk.astype(np.float32)


# ============================================================================
# VERIFICACIÓN DE INTEGRIDAD (ejecutar una vez antes de entrenar)
# ============================================================================

def verificar_splits(splits_dir=None, esperar_gt=True):
    """
    Comprobación previa al entrenamiento.

    Verifica que los splits existen, que train/val cargan, y que el test
    cumple la condición D1-R1 (GT plata presente) si se pidió.
    """
    splits_dir = splits_dir or SPLITS_DIR
    print("=" * 74)
    print("VERIFICACIÓN PREVIA DE SPLITS")
    print("=" * 74)

    informe = {}
    for split in ("train", "val", "test_pos", "test_neg"):
        try:
            ds = DatasetTB(split, splits_dir=splits_dir,
                           esperar_gt=esperar_gt, verbose=False)
            x, m, info = ds[0]
            areas = []
            for j in range(min(len(ds), 8)):   # muestra las primeras 8
                _, mm, _ = ds[j]
                areas.append(float(mm.mean()))
            informe[split] = {
                "n": len(ds),
                "gt": ds.fuente_gt,
                "imagen": tuple(x.shape),
                "mascara": tuple(m.shape),
                "area_media_muestra": round(float(np.mean(areas)), 5),
                "estado": "OK",
            }
            print(f"  [OK ] {split:9s} n={len(ds):3d}  "
                  f"img={tuple(x.shape)}  GT={ds.fuente_gt}  "
                  f"área(muestra)={np.mean(areas):.4f}")
        except Exception as e:
            informe[split] = {"estado": "ERROR", "detalle": str(e)}
            print(f"  [!!] {split:9s} {type(e).__name__}: {str(e).splitlines()[0]}")

    ok = all(v.get("estado") == "OK" for v in informe.values())
    print("-" * 74)
    if ok:
        print("Splits íntegros. Listo para entrenar.")
    else:
        print("Hay splits con problemas. Resolver ANTES de entrenar.")
    return informe


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Verifica los splits sellados.")
    p.add_argument("--sin-gt", action="store_true",
                   help="No exige el GT del test (solo para probar la carga)")
    p.add_argument("--splits-dir", default=None)
    a = p.parse_args()
    verificar_splits(splits_dir=a.splits_dir, esperar_gt=not a.sin_gt)
