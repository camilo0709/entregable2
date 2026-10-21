"""
eda_lib_tb.py — utilidades del EDA orientado a decisiones (Entregable 2, §2).

Usado por libro/book/EDA_Decisiones_TBX11K.ipynb. Todas las funciones son
deterministas (semilla 42). Convenciones del proyecto:

- La membresía de un caso la define el CSV del split, nunca el directorio
  (§R1.8 de docs/DECISION_D1_GROUND_TRUTH.md); `ruta_local` remapea las rutas
  Windows de los CSV a la raíz local.
- El GT del test_pos es plata SAM 2 congelada (D1-R1); aquí solo se mide,
  nunca se modifica.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SEMILLA = 42


# --------------------------------------------------------------------------- #
# Rutas y splits
# --------------------------------------------------------------------------- #
def encontrar_root(inicio: str | Path | None = None) -> Path:
    """Raíz del proyecto: el primer directorio hacia arriba que contiene
    datos/splits/. Ejecutar el notebook desde la raíz del proyecto."""
    p = Path(inicio or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "datos" / "splits").is_dir():
            return cand
    raise FileNotFoundError("No encuentro la raíz del proyecto (datos/splits/)")


def cargar_splits(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for nombre in ("train", "val", "test_pos", "test_neg", "test_pos_reserva"):
        out[nombre] = pd.read_csv(root / "datos" / "splits" / f"{nombre}.csv")
    return out


def ruta_local(ruta_windows: str, root: Path) -> Path:
    """C:\\Users\\...\\proyecto_tb\\datos\\x -> <root>/datos/x (§R1.8)."""
    norm = ruta_windows.replace("\\", "/")
    i = norm.find("datos/")
    if i < 0:
        raise ValueError(f"ruta sin prefijo datos/: {ruta_windows}")
    return root / norm[i:]


# --------------------------------------------------------------------------- #
# Duplicados exactos y casi-duplicados (§2.1 Calidad)
# --------------------------------------------------------------------------- #
def md5_file(path: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bloque), b""):
            h.update(b)
    return h.hexdigest()


def phash64(img: Image.Image) -> int:
    """pHash de 64 bits (DCT sobre 32×32 en escala de grises; umbral = mediana
    de los 64 coeficientes de baja frecuencia, sin DC)."""
    from scipy.fftpack import dct

    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=float)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    d[0, 0] = 0.0  # el componente DC no codifica estructura
    bits = (d >= np.median(d)).flatten().astype(np.int64)
    return int("".join(map(str, bits.tolist())), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Estadísticos de apariencia (tests de shortcut / distribución, §2.1-2.2)
# --------------------------------------------------------------------------- #
def features_bajo_nivel(img_u8: np.ndarray) -> dict:
    """Apariencia (NO contenido): separa train de test si hay sesgo de
    adquisición. Un clasificador sobre estas features con AUC >> 0.5 implica
    shortcut de adquisición."""
    x = img_u8.astype(np.float32)
    p5, p95 = np.percentile(x, 5), np.percentile(x, 95)
    gx = np.abs(np.diff(x, axis=1)).mean()
    gy = np.abs(np.diff(x, axis=0)).mean()
    return {
        "media": float(x.mean()),
        "desv": float(x.std()),
        "p5": float(p5),
        "p95": float(p95),
        "contraste": float(p95 - p5),
        "sat_claro": float((x >= 250).mean()),
        "sat_oscuro": float((x <= 5).mean()),
        "gradiente": float(gx + gy),
    }


def auc_cv_permutacion(X, y, semilla: int = SEMILLA, cv: int = 5, B: int = 200):
    """AUC media por CV estratificada de regresión logística sobre X y test de
    permutación (H0: AUC = 0.5). Devuelve (auc, p_perm)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = np.asarray(X, float)
    y = np.asarray(y, int)

    def auc_de(yy):
        m = LogisticRegression(max_iter=3000)
        skf = StratifiedKFold(cv, shuffle=True, random_state=semilla)
        return float(cross_val_score(m, X, yy, cv=skf, scoring="roc_auc").mean())

    obs = auc_de(y)
    rng = np.random.RandomState(semilla)
    ge = sum(auc_de(rng.permutation(y)) >= obs for _ in range(B))
    return obs, (1 + ge) / (B + 1)


# --------------------------------------------------------------------------- #
# Visualización
# --------------------------------------------------------------------------- #
def overlay_mask(img_u8: np.ndarray, mask_bool: np.ndarray,
                 color=(255, 60, 60), alpha: float = 0.45) -> Image.Image:
    """Radiografía en gris + máscara translúcida con contorno."""
    import cv2

    rgb = np.stack([img_u8] * 3, -1).astype(np.float32)
    m = mask_bool.astype(bool)
    rgb[m] = rgb[m] * (1 - alpha) + np.array(color, np.float32) * alpha
    borde = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT,
                             np.ones((3, 3), np.uint8)).astype(bool)
    rgb[borde] = color
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def montaje(imagenes, etiquetas, ncol: int = 4, tam: int = 180,
            ruta: Path | None = None) -> Image.Image:
    """Montaje con etiquetas debajo de cada celda."""
    n = len(imagenes)
    nrow = (n + ncol - 1) // ncol
    lienzo = Image.new("RGB", (ncol * tam, nrow * (tam + 16)), "white")
    d = ImageDraw.Draw(lienzo)
    for i, (im, et) in enumerate(zip(imagenes, etiquetas)):
        cx, cy = (i % ncol) * tam, (i // ncol) * (tam + 16)
        if not isinstance(im, Image.Image):
            im = Image.fromarray(np.asarray(im))
        lienzo.paste(im.convert("RGB").resize((tam, tam), Image.LANCZOS), (cx, cy))
        d.text((cx + 4, cy + tam + 3), str(et)[:24], fill="black")
    if ruta:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        lienzo.save(ruta)
    return lienzo


def cargar_png_gray_u8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def guardar_fig(fig, root: Path, nombre: str) -> Path:
    d = root / "libro" / "book" / "figs_eda"
    d.mkdir(parents=True, exist_ok=True)
    p = d / nombre
    fig.savefig(p, dpi=110, bbox_inches="tight")
    return p
