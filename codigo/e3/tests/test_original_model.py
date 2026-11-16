"""
PRUEBAS UNITARIAS DEL MODELO ORIGINAL (LACCA) - Entregable 3, Bloque C.

Cubre las seis pruebas del enunciado (sobreajuste de lote pequeño,
reproducibilidad, formas y entradas inválidas, flujo de gradientes,
propiedades teóricas, guardar/cargar) más las propias del mecanismo:
equivarianza al volteo, identidad con gamma=0, reflexión exacta con el eje
en la línea media, eje acotado y asimetría nula en entrada simétrica.

Ejecutar:  python -m pytest codigo/tests -q
"""

import io
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from original_model import LACCA, TinySegNet, UNetLACCA  # noqa: E402


def lote(n=4, c=1, h=16, w=16, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    x = torch.randn(n, c, h, w, generator=g)
    y = (x > 0).float()
    return x, y


def lacca_aleatorio(canales=6, dtype=torch.float64, **kw):
    """LACCA con eje, sesgo y compuerta no triviales (para propiedades)."""
    torch.manual_seed(1)
    args = dict(key_dim=4, value_dim=5, radius=2, gamma_init=0.7,
                axis_init_std=0.8)
    args.update(kw)
    m = LACCA(canales, **args).to(dtype)
    with torch.no_grad():
        m.sesgo.normal_(0.0, 0.5)
    return m.eval()


# ---------------------------------------------------------------- prueba 1
def test_sobreajusta_lote_pequeno():
    x, y = lote()
    modelo = TinySegNet(semilla=0)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-2)
    for _ in range(250):
        opt.zero_grad()
        perdida = torch.nn.functional.binary_cross_entropy_with_logits(
            modelo(x), y)
        perdida.backward()
        opt.step()
    assert perdida.item() < 0.1


# ---------------------------------------------------------------- prueba 2
def test_reproducible_con_misma_semilla():
    x, _ = lote()
    p1 = TinySegNet(semilla=1).predict_proba(x)
    p2 = TinySegNet(semilla=1).predict_proba(x)
    torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------- prueba 3
def test_formas_y_entradas_invalidas():
    m = lacca_aleatorio(canales=6, dtype=torch.float32)
    x = torch.randn(2, 6, 12, 10)                      # no cuadrada a propósito
    assert m(x).shape == x.shape
    salida, theta, d = m(x, devolver_eje=True)
    assert theta.shape == (2,) and d.shape == (2,)
    with pytest.raises(RuntimeError):
        m(torch.randn(2, 5, 12, 10))                   # canales equivocados
    with pytest.raises(ValueError):
        m(torch.randn(6, 12, 10))                      # sin dimensión de lote
    with pytest.raises(RuntimeError):
        TinySegNet()(torch.randn(4, 7, 16, 16))        # 7 canales, espera 1


# ---------------------------------------------------------------- prueba 4
def test_todos_los_parametros_reciben_gradiente():
    x, y = lote()
    modelo = TinySegNet(semilla=0)
    torch.nn.functional.binary_cross_entropy_with_logits(
        modelo(x), y).backward()
    for nombre, p in modelo.named_parameters():
        assert p.grad is not None, nombre
        assert torch.isfinite(p.grad).all(), nombre
        assert p.grad.abs().sum() > 0, f"gradiente nulo en {nombre}"


# ---------------------------------------------------------------- prueba 5
def test_propiedad_equivarianza_al_volteo():
    """LACCA(F∘S) == LACCA(F)∘S  (Proposición 1; float64)."""
    m = lacca_aleatorio()
    torch.manual_seed(3)
    for h, w in [(14, 14), (12, 18)]:
        f = torch.randn(3, 6, h, w, dtype=torch.float64)
        a = m(torch.flip(f, dims=(3,)))
        b = torch.flip(m(f), dims=(3,))
        assert (a - b).abs().max().item() < 1e-8


def test_eje_cambia_de_signo_al_voltear():
    m = lacca_aleatorio()
    f = torch.randn(3, 6, 14, 14, dtype=torch.float64)
    t1, d1 = m.eje(f)
    t2, d2 = m.eje(torch.flip(f, dims=(3,)))
    assert torch.allclose(t2, -t1, atol=1e-10)
    assert torch.allclose(d2, -d1, atol=1e-10)


def test_identidad_con_gamma_cero():
    """Proposición 3: con gamma=0 la salida es exactamente la entrada."""
    m = lacca_aleatorio(gamma_init=0.0)
    f = torch.randn(2, 6, 10, 10, dtype=torch.float64)
    assert torch.equal(m(f), f)


def test_reflexion_exacta_con_eje_en_linea_media():
    """Con eje (theta=0, d=0) el espejo es el volteo horizontal."""
    m = lacca_aleatorio()
    f = torch.randn(2, 6, 10, 12, dtype=torch.float64)
    z = torch.zeros(2, dtype=torch.float64)
    espejo, mascara = m._reflejar(f, z, z)
    assert torch.allclose(espejo, torch.flip(f, dims=(3,)), atol=1e-12)
    assert torch.all(mascara == 1)


def test_eje_acotado():
    m = lacca_aleatorio(theta_max=math.radians(10), d_max=0.1)
    f = 5.0 * torch.randn(8, 6, 14, 14, dtype=torch.float64)
    theta, d = m.eje(f)
    assert theta.abs().max() <= math.radians(10) + 1e-12
    assert d.abs().max() <= 0.1 + 1e-12


def test_asimetria_nula_en_entrada_simetrica():
    """Con radio 0 y entrada simétrica, D = W_v F - W_v F~ = 0 (eje central)."""
    m = lacca_aleatorio(radius=0, axis_init_std=0.0)
    mitad = torch.randn(2, 6, 10, 6, dtype=torch.float64)
    f = torch.cat((mitad, torch.flip(mitad, dims=(3,))), dim=3)
    D, theta, d = m.asimetria(f)
    assert theta.abs().max() < 1e-12 and d.abs().max() < 1e-12
    assert D.abs().max().item() < 1e-10


# ---------------------------------------------------------------- prueba 5b
def test_sensibilidad_acotada_empirica():
    """Comprobación EMPÍRICA (no es la demostración): el cociente
    ||M(F+e) - M(F)|| / ||e|| se mantiene finito y moderado aun con
    entradas de escala grande, gracias a la normalización con epsilon."""
    m = lacca_aleatorio()
    torch.manual_seed(5)
    razones = []
    for escala in (1.0, 10.0, 100.0):
        f = escala * torch.randn(2, 6, 12, 12, dtype=torch.float64)
        for _ in range(5):
            e = 1e-6 * torch.randn_like(f)
            razon = (m(f + e) - m(f)).norm() / e.norm()
            razones.append(razon.item())
    assert all(math.isfinite(r) for r in razones)
    assert max(razones) < 1e3


# ---------------------------------------------------------------- prueba 6
def test_guardar_y_cargar_conserva_salidas():
    x, _ = lote()
    m1 = TinySegNet(semilla=1)
    m2 = TinySegNet(semilla=99)
    buf = io.BytesIO()
    torch.save(m1.state_dict(), buf)
    buf.seek(0)
    m2.load_state_dict(torch.load(buf))
    torch.testing.assert_close(m1.predict_proba(x), m2.predict_proba(x))


# ------------------------------------------------------- modelo completo
def test_unet_lacca_forma_y_gradiente():
    pytest.importorskip("segmentation_models_pytorch")
    torch.manual_seed(0)
    m = UNetLACCA(etapas=(3, 4), key_dim=4, value_dim=8, radius=1,
                  gamma_init=0.1, axis_init_std=0.1)
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    y.mean().backward()
    for nombre, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), nombre


def test_unet_lacca_sin_etapas_es_la_unet_base():
    pytest.importorskip("segmentation_models_pytorch")
    m = UNetLACCA(etapas=())
    assert len(m.base.encoder.lacca) == 0
    assert m(torch.randn(1, 1, 64, 64)).shape == (1, 1, 64, 64)


# ------------------------------------------- pruebas añadidas tras mutación
def test_asimetria_cero_donde_el_espejo_cae_fuera():
    """Con eje desplazado, las columnas sin contraparte no generan asimetría."""
    m = lacca_aleatorio()
    m.eje = lambda f: (torch.zeros(f.shape[0], dtype=f.dtype),
                       torch.full((f.shape[0],), 0.25, dtype=f.dtype))
    f = torch.randn(2, 6, 12, 16, dtype=torch.float64)
    D, _, _ = m.asimetria(f)
    assert D[..., :3].abs().max().item() == 0.0      # x < -0.5: sin espejo
    assert D[..., 8:].abs().max().item() > 0         # zona válida sí cambia


def test_normalizacion_acota_q_y_k():
    from original_model import _normalizar
    x = 1e4 * torch.randn(3, 8, 5, 5, dtype=torch.float64)
    assert _normalizar(x, 1e-3).norm(dim=1).max().item() <= 1.0 + 1e-12


def test_exporta_a_onnx_y_coincide(tmp_path):
    """Ida y vuelta ONNX del módulo (tolerancia 1e-4 del enunciado)."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    m = lacca_aleatorio(canales=8, dtype=torch.float32)
    x = torch.randn(2, 8, 16, 20)
    f = str(tmp_path / "lacca.onnx")
    torch.onnx.export(m, (x,), f, opset_version=17, dynamo=False,
                      input_names=["x"], output_names=["y"])
    s = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
    a = s.run(None, {"x": x.numpy()})[0]
    with torch.no_grad():
        b = m(x).numpy()
    assert abs(a - b).max() < 1e-4


def test_cota_lipschitz_puntual_se_cumple():
    """Eje fijo: ||dF'_i|| <= [1+|g|(|psi_f| + |psi_D|(2|Wv| + K2|Wv|R(cq+ck)/(2tau)))] d."""
    r, C, eps, tau, R = 1, 6, 1e-3, 0.1, 5.0
    m = lacca_aleatorio(canales=C, radius=r, eps=eps, tau=tau)
    m.eje = lambda f: (torch.full((f.shape[0],), 0.05, dtype=f.dtype),
                       torch.full((f.shape[0],), 0.04, dtype=f.dtype))
    sn = lambda w: torch.linalg.matrix_norm(w.flatten(1), 2).item()
    K2 = (2 * r + 1) ** 2
    Wv = sn(m.wv.weight)
    cq, ck = sn(m.wq.weight) / math.sqrt(eps), sn(m.wk.weight) / math.sqrt(eps)
    P = m.psi.weight.squeeze(-1).squeeze(-1)
    LA = Wv + 0.5 * K2 * Wv * R * (cq + ck) / tau
    cota = 1 + abs(m.gamma.item()) * (sn(P[:, :C]) + sn(P[:, C:]) * (Wv + LA))
    torch.manual_seed(0)
    f = torch.randn(1, C, 10, 10, dtype=torch.float64)
    f = f * (R / f.norm(dim=1, keepdim=True).max())
    for _ in range(10):
        e = torch.randn_like(f)
        e = 1e-6 * e / e.norm(dim=1, keepdim=True).max()
        razon = (m(f + e) - m(f)).norm(dim=1).max() / 1e-6
        assert razon.item() <= cota


# ------------------------------------------------------- ablaciones (flags)
@pytest.mark.parametrize("flags", [
    {"axis_mode": "fixed"}, {"asym_mode": "direct"}, {"attn_mode": "global"},
    {"axis_mode": "fixed", "asym_mode": "direct", "attn_mode": "global"}])
def test_ablaciones_forma_y_equivarianza(flags):
    m = lacca_aleatorio(**flags)
    f = torch.randn(2, 6, 12, 14, dtype=torch.float64)
    a = m(torch.flip(f, dims=(3,)))
    b = torch.flip(m(f), dims=(3,))
    assert m(f).shape == f.shape
    assert (a - b).abs().max().item() < 1e-8, flags


def test_ablacion_eje_fijo_no_depende_del_mapa_de_calor():
    m = lacca_aleatorio(axis_mode="fixed")
    t, d = m.eje(torch.randn(2, 6, 10, 10, dtype=torch.float64))
    assert torch.all(t == 0) and torch.all(d == 0)


def test_ablacion_global_atiende_mas_alla_de_la_ventana():
    """Con atención global, perturbar un píxel lejano cambia la salida."""
    m = lacca_aleatorio(attn_mode="global", radius=1)
    z = torch.zeros(2, dtype=torch.float64)
    m.eje = lambda f: (torch.zeros(f.shape[0], dtype=f.dtype),
                       torch.zeros(f.shape[0], dtype=f.dtype))
    f = torch.randn(1, 6, 12, 12, dtype=torch.float64)
    g = f.clone()
    g[0, :, 6, 5] += 1.0      # su espejo (6,6) queda fuera de la ventana de (6,0)
    d = (m(g) - m(f))[0, :, 6, 0].abs().max().item()
    ml = lacca_aleatorio(attn_mode="local", radius=1)
    ml.eje = m.eje
    dl = (ml(g) - ml(f))[0, :, 6, 0].abs().max().item()
    assert d > 1e-9 and dl < 1e-12


def test_flags_invalidos():
    with pytest.raises(ValueError):
        LACCA(4, axis_mode="otro")
    with pytest.raises(ValueError):
        LACCA(4, attn_mode="otro")
