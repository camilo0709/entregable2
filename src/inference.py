"""Inferencia ONNX en CPU. Carga bajo demanda y como máximo 1 modelo en memoria."""
import hashlib, os, tempfile, threading, urllib.request
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image

from .data import ROOT, perfil

TAM = 512
MAX_MB = 8
FORMATOS = {"PNG", "JPEG", "BMP", "TIFF"}
# Carpeta local con los .onnx y, si no están, URL base de los pesos (assets de un Release).
DIR_PESOS = Path(os.environ.get("TB_WEIGHTS_DIR", ROOT / "artifacts" / "onnx"))
URL_BASE = os.environ.get("TB_WEIGHTS_URL",
                          "https://github.com/camilo0709/entregable3/releases/download/pesos-onnx-v1")
MAX_EN_MEMORIA = 1
LIMITE_PARAMS = 60_000_000  # por encima, solo resultados precalculados en el plan gratuito
_cache: "OrderedDict[str, object]" = OrderedDict()
_lock = threading.Lock()


class EntradaInvalida(ValueError):
    pass


def validar_y_preparar(raw: bytes) -> np.ndarray:
    """Valida la imagen subida y devuelve un arreglo float32 [512,512] en [0,255]."""
    if len(raw) > MAX_MB * 1024 * 1024:
        raise EntradaInvalida(f"El archivo supera {MAX_MB} MB.")
    try:
        import io
        im = Image.open(io.BytesIO(raw)); im.load()
    except Exception:
        raise EntradaInvalida("No se pudo leer la imagen. Use PNG, JPEG, BMP o TIFF.")
    if (im.format or "").upper() not in FORMATOS:
        raise EntradaInvalida("Formato no admitido. Use PNG, JPEG, BMP o TIFF.")
    if min(im.size) < 128:
        raise EntradaInvalida("La imagen es demasiado pequeña (mínimo 128 px por lado).")
    if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        # radiografías de 16 bits: escalar por percentiles a 0-255 (convert("L") las satura)
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        im = Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8))
    im = im.convert("L").resize((TAM, TAM), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def normalizar(img: np.ndarray) -> np.ndarray:
    """z-score por imagen, igual que en el entrenamiento."""
    return ((img - img.mean()) / (img.std() + 1e-6)).astype(np.float32)


def _ruta_modelo(nombre: str) -> Path:
    p = DIR_PESOS / f"{nombre}.onnx"
    if p.exists():
        return p
    DIR_PESOS.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"{nombre}.onnx"
    if not tmp.exists():
        urllib.request.urlretrieve(f"{URL_BASE}/{nombre}.onnx", tmp)
    esperado = perfil()[nombre]["sha256_onnx"]
    h = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if h != esperado:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("El archivo del modelo no coincide con el SHA-256 registrado.")
    return tmp


def _sesion(nombre: str):
    import onnxruntime as ort
    with _lock:
        if nombre in _cache:
            _cache.move_to_end(nombre)
            return _cache[nombre]
        while len(_cache) >= MAX_EN_MEMORIA:
            _cache.popitem(last=False)
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        s = ort.InferenceSession(str(_ruta_modelo(nombre)), so, providers=["CPUExecutionProvider"])
        _cache[nombre] = s
        return s


def predecir(nombre: str, img: np.ndarray, umbral: float = 0.5, T: float = 1.0):
    """Devuelve (prob[512,512], mascara bool, incertidumbre[512,512], latencia_ms)."""
    import time
    if nombre not in perfil():
        raise EntradaInvalida("Modelo desconocido.")
    if perfil()[nombre]["parametros"] > LIMITE_PARAMS and os.environ.get("TB_PERMITIR_PESADOS") != "1":
        raise EntradaInvalida("Este modelo (UNETR, 88 M parámetros) no cabe en la memoria del plan gratuito de Render; "
                              "sus resultados precalculados están en la tabla y en el diagnóstico.")
    s = _sesion(nombre)
    x = normalizar(img)[None, None]
    t0 = time.perf_counter()
    logits = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    ms = (time.perf_counter() - t0) * 1000
    prob = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-3)))
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    incert = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))  # entropía binaria en bits
    return prob.astype(np.float32), prob >= umbral, incert.astype(np.float32), ms


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum()); s = float(pred.sum() + gt.sum())
    return 1.0 if s == 0 else 2 * inter / s
