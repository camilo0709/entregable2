"""Pruebas del tablero (no requieren los pesos ONNX completos)."""
import io, os, sys
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import inference as inf, data, plots


def _png(w=600, h=600, fmt="PNG"):
    b = io.BytesIO(); Image.fromarray((np.random.rand(h, w) * 255).astype("uint8")).save(b, fmt); return b.getvalue()


def test_validar_ok():
    x = inf.validar_y_preparar(_png())
    assert x.shape == (512, 512) and x.dtype == np.float32


def test_validar_rechaza_basura_pequena_y_grande():
    with pytest.raises(inf.EntradaInvalida): inf.validar_y_preparar(b"no es una imagen")
    with pytest.raises(inf.EntradaInvalida): inf.validar_y_preparar(_png(64, 64))
    with pytest.raises(inf.EntradaInvalida): inf.validar_y_preparar(b"0" * (inf.MAX_MB * 1024 * 1024 + 1))


def test_normalizar_zscore():
    z = inf.normalizar(np.random.rand(512, 512).astype("float32") * 200)
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3


def test_dice():
    a = np.zeros((8, 8), bool); a[:4] = True
    assert inf.dice(a, a) == 1.0 and inf.dice(a, ~a) == 0.0 and inf.dice(~a & False, ~a & False) == 1.0


def test_modelo_desconocido():
    with pytest.raises(inf.EntradaInvalida): inf.predecir("no_existe", np.zeros((512, 512), "float32"))


def test_tablas_y_figuras():
    lb = data.leaderboard()
    assert len(lb) == 14 and lb.dice_media.between(0, 1).all()
    assert plots.barras_dice(lb) is not None and plots.pareto(lb) is not None
    assert set(data.perfil()) == set(lb.modelo)


def test_paginas_registradas():
    import app, dash
    assert {p["path"] for p in dash.page_registry.values()} == {"/", "/eda", "/modelos"}


def test_png_16_bits_no_se_satura():
    a = (np.random.rand(600, 600) * 60000).astype("uint16")
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
    x = inf.validar_y_preparar(b.getvalue())
    assert x.std() > 30 and x.min() >= 0 and x.max() <= 255
