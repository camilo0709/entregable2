"""
================================================================================
MODELO ORIGINAL - LACCA (Learned-Axis Contralateral Cross-Attention)
================================================================================

PROPÓSITO:
    Módulo propio que (1) estima por imagen un eje de simetría torácica,
    (2) refleja el mapa de características respecto a ese eje, (3) deja que
    cada posición atienda a una ventana local de su espejo y (4) inyecta en la
    red la ASIMETRÍA local (lo que difiere de la contraparte) mediante una
    compuerta que parte en cero.

    Diseño y fórmulas: doc "Entregable 3 TB", sección "Diseño del módulo de
    simetría (borrador v0)".

PIEZAS PERMITIDAS COMO BLOQUES DE CONSTRUCCIÓN (enunciado, R2):
    nn.Conv2d (1x1), softmax, grid_sample, pad/slice. El MECANISMO (eje
    antisimetrizado, reflexión, atención local al espejo, residual de
    asimetría con compuerta) está programado aquí.

CONTENIDO:
    LACCA           módulo propio (nn.Module) sobre mapas de características
    TinySegNet      red mínima para pruebas unitarias
    UNetLACCA       U-Net (smp, resnet34, sin preentrenar) con LACCA en las
                    conexiones de salto elegidas; es el "modelo original"
    ModeloOriginal  alias de UNetLACCA

NOTAS DE DISEÑO (diferencias menores frente al borrador v0):
    - La señal de asimetría es D = W_v F(x) - A(x): se proyecta F con el
      mismo W_v que los valores del espejo (D es una función lineal de la
      diferencia F - F_espejo ponderada). Así D vive en dim 'value_dim' y la
      memoria no depende de C.
    - Las ventanas locales se construyen con pad + slicing (no F.unfold)
      para que el módulo sea exportable a ONNX sin operadores raros.
    - Con gamma_init=0 (por diseño) solo gamma recibe gradiente en el primer
      paso (ReZero); las pruebas de gradiente usan gamma_init > 0.
================================================================================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


# ============================================================================
# UTILIDADES
# ============================================================================

def _ventanas(x, r):
    """(B,C,H,W) -> (B,C,K*K,H,W), K=2r+1, con relleno de ceros.

    El índice de ventana j = iy*K + ix corresponde al desplazamiento
    (dx, dy) = (ix - r, iy - r).
    """
    _, _, h, w = x.shape
    xp = Fn.pad(x, (r, r, r, r))
    piezas = []
    for iy in range(2 * r + 1):
        for ix in range(2 * r + 1):
            piezas.append(xp[:, :, iy:iy + h, ix:ix + w])
    return torch.stack(piezas, dim=2)


def _normalizar(x, eps):
    """x / sqrt(||x||^2 + eps) sobre el eje de canales (acota el Jacobiano)."""
    return x / torch.sqrt((x * x).sum(dim=1, keepdim=True) + eps)


# ============================================================================
# MÓDULO PROPIO
# ============================================================================

class LACCA(nn.Module):
    """Atención cruzada contralateral con eje de reflexión aprendido.

    Entrada y salida: tensores (B, C, H, W). La salida es
        F' = F + gamma * psi([F, D]).
    """

    def __init__(self, channels, key_dim=16, value_dim=32, radius=2,
                 theta_max=math.radians(15.0), d_max=0.25, eps=1e-3,
                 tau=0.1, gamma_init=0.0, axis_init_std=0.0,
                 axis_mode="learned", asym_mode="diff", attn_mode="local"):
        super().__init__()
        # flags de ABLACIÓN (por defecto = modelo completo):
        #   axis_mode "fixed"   eje en la línea media (theta=d=0)
        #   asym_mode "direct"  D = A (sin restar la propia señal W_v F)
        #   attn_mode "global"  atención sobre todas las posiciones del espejo
        #                       (sin sesgo posicional; coste O((HW)^2))
        if axis_mode not in ("learned", "fixed"):
            raise ValueError("axis_mode: 'learned' | 'fixed'")
        if asym_mode not in ("diff", "direct"):
            raise ValueError("asym_mode: 'diff' | 'direct'")
        if attn_mode not in ("local", "global"):
            raise ValueError("attn_mode: 'local' | 'global'")
        self.axis_mode, self.asym_mode, self.attn_mode = (
            axis_mode, asym_mode, attn_mode)
        if channels < 1 or key_dim < 1 or value_dim < 1 or radius < 0:
            raise ValueError("channels, key_dim, value_dim >= 1 y radius >= 0")
        if eps <= 0 or tau <= 0 or theta_max <= 0 or d_max <= 0:
            raise ValueError("eps, tau, theta_max y d_max deben ser > 0")
        self.channels = channels
        self.radius = radius
        self.theta_max = float(theta_max)
        self.d_max = float(d_max)
        self.eps = float(eps)
        self.tau = float(tau)
        k = 2 * radius + 1
        self.k = k

        # --- cabeza del eje: mapa de calor 1x1 (sin sesgo: invariante a
        #     desplazamientos del softmax, su gradiente sería nulo) ---
        self.mapa = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        nn.init.normal_(self.mapa.weight, mean=0.0, std=float(axis_init_std))

        # --- proyecciones de atención (sin sesgo) ---
        self.wq = nn.Conv2d(channels, key_dim, kernel_size=1, bias=False)
        self.wk = nn.Conv2d(channels, key_dim, kernel_size=1, bias=False)
        self.wv = nn.Conv2d(channels, value_dim, kernel_size=1, bias=False)

        # --- sesgo posicional simétrico: b(dx,dy) = b(-dx,dy) ---
        self.sesgo = nn.Parameter(torch.zeros(k, radius + 1))
        idx = (torch.arange(k) - radius).abs()
        self.register_buffer("_idx_dx", idx, persistent=False)

        # --- mezcla y compuerta ---
        self.psi = nn.Conv2d(channels + value_dim, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    # ------------------------------------------------------------------ eje
    @staticmethod
    def _coords(h, w, dtype, device):
        """Coordenadas de centros de píxel en [-1,1] y escalas isotrópicas."""
        xn = (2.0 * torch.arange(w, dtype=dtype, device=device) + 1.0) / w - 1.0
        yn = (2.0 * torch.arange(h, dtype=dtype, device=device) + 1.0) / h - 1.0
        m = float(max(h, w))
        return xn, yn, w / m, h / m   # xn, yn, sx, sy

    def _u(self, f):
        """Cabeza del eje sobre un mapa: (B,2) = (u1 ~ inclinación, u2 ~ x)."""
        b, _, h, w = f.shape
        xn, yn, sx, sy = self._coords(h, w, f.dtype, f.device)
        ux = (xn * sx).view(1, 1, 1, w)
        uy = (yn * sy).view(1, 1, h, 1)
        p = torch.softmax(self.mapa(f).flatten(1), dim=1).view(b, 1, h, w)
        cx = (p * ux).sum(dim=(1, 2, 3))
        cy = (p * uy).sum(dim=(1, 2, 3))
        dx = ux - cx.view(b, 1, 1, 1)
        dy = uy - cy.view(b, 1, 1, 1)
        cov = (p * dx * dy).sum(dim=(1, 2, 3))
        vary = (p * dy * dy).sum(dim=(1, 2, 3)) + 1e-6
        pendiente = cov / vary                    # dx/dy de la recta
        return torch.stack((-torch.atan(pendiente), cx), dim=1)

    def eje(self, f):
        """(theta, d) por imagen, EQUIVARIANTES al volteo por construcción.

        Antisimetrización (promedio sobre el grupo {1, S}):
            hat = 1/2 (u(F) - u(F∘S));  theta = theta_max tanh(hat/theta_max).
        """
        if self.axis_mode == "fixed":
            z = torch.zeros(f.shape[0], dtype=f.dtype, device=f.device)
            return z, z.clone()
        u = self._u(f)
        us = self._u(torch.flip(f, dims=(3,)))
        hat = 0.5 * (u - us)
        theta = self.theta_max * torch.tanh(hat[:, 0] / self.theta_max)
        d = self.d_max * torch.tanh(hat[:, 1] / self.d_max)
        return theta, d

    # ----------------------------------------------------------- reflexión
    def _reflejar(self, f, theta, d):
        """Mapa espejo F~(x) = F(R(x)) y máscara de validez M(x)."""
        b, _, h, w = f.shape
        xn, yn, sx, sy = self._coords(h, w, f.dtype, f.device)
        ux = (xn * sx).view(1, 1, w)
        uy = (yn * sy).view(1, h, 1)
        nx = torch.cos(theta).view(b, 1, 1)
        ny = torch.sin(theta).view(b, 1, 1)
        dist = nx * ux + ny * uy - d.view(b, 1, 1)         # (B,H,W)
        gx = (ux - 2.0 * dist * nx) / sx
        gy = (uy - 2.0 * dist * ny) / sy
        grid = torch.stack((gx, gy), dim=-1)               # (B,H,W,2) (x,y)
        mask = ((gx.abs() <= 1.0) & (gy.abs() <= 1.0)).to(f.dtype).unsqueeze(1)
        espejo = Fn.grid_sample(f, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        return espejo * mask, mask

    # ------------------------------------------------------------- atención
    def _sesgo_plano(self):
        return self.sesgo[:, self._idx_dx].reshape(-1)      # (K*K,)

    def _calcular(self, f):
        if f.dim() != 4:
            raise ValueError(f"se esperaba (B,C,H,W), llegó {tuple(f.shape)}")
        if f.shape[1] != self.channels:
            raise RuntimeError(
                f"LACCA espera {self.channels} canales, llegaron {f.shape[1]}")
        theta, d = self.eje(f)
        espejo, m = self._reflejar(f, theta, d)

        q = _normalizar(self.wq(f), self.eps)
        kk = _normalizar(self.wk(espejo), self.eps)
        v = self.wv(espejo)

        if self.attn_mode == "global":
            b, _, h, w = f.shape
            qf, kf, vf = q.flatten(2), kk.flatten(2), v.flatten(2)
            sg = torch.bmm(qf.transpose(1, 2), kf) / self.tau       # (B,N,N)
            sg = sg + (m.flatten(2) - 1.0) * 1e4                    # claves
            ag = torch.softmax(sg, dim=-1)
            A = torch.bmm(vf, ag.transpose(1, 2)).view(b, -1, h, w)
        else:
            kw = _ventanas(kk, self.radius)                 # (B,dk,K2,H,W)
            vw = _ventanas(v, self.radius)                  # (B,dv,K2,H,W)
            mw = _ventanas(m, self.radius)[:, 0]            # (B,K2,H,W)

            s = (q.unsqueeze(2) * kw).sum(dim=1) / self.tau
            s = s + self._sesgo_plano().view(1, -1, 1, 1)
            s = s + (mw - 1.0) * 1e4                        # enmascara
            a = torch.softmax(s, dim=1)
            A = (a.unsqueeze(1) * vw).sum(dim=2)            # (B,dv,H,W)

        if self.asym_mode == "direct":
            D = A * m
        else:
            D = (self.wv(f) - A) * m                        # señal de asimetría
        return D, theta, d

    def asimetria(self, f):
        """Devuelve (D, theta, d): señal de asimetría y eje estimado."""
        return self._calcular(f)

    def forward(self, f, devolver_eje=False):
        D, theta, d = self._calcular(f)
        salida = f + self.gamma * self.psi(torch.cat((f, D), dim=1))
        if devolver_eje:
            return salida, theta, d
        return salida


# ============================================================================
# REDES
# ============================================================================

class TinySegNet(nn.Module):
    """Red mínima conv -> LACCA -> conv para pruebas unitarias y humo."""

    def __init__(self, in_channels=1, hidden=8, semilla=42, **kw_lacca):
        super().__init__()
        torch.manual_seed(semilla)
        kw = dict(key_dim=4, value_dim=8, radius=1, gamma_init=0.1,
                  axis_init_std=0.1)
        kw.update(kw_lacca)
        self.entrada = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.lacca = LACCA(hidden, **kw)
        self.salida = nn.Conv2d(hidden, 1, 1)
        self.in_channels = in_channels

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"entrada inválida {tuple(x.shape)}; se espera "
                f"(B,{self.in_channels},H,W)")
        h = torch.relu(self.entrada(x))
        h = self.lacca(h)
        return self.salida(h)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.sigmoid(self.forward(x))


class _EncoderConLACCA(nn.Module):
    """Envuelve un encoder de smp y aplica LACCA a las etapas elegidas."""

    def __init__(self, encoder, lacca_por_etapa):
        super().__init__()
        self.enc = encoder
        self.lacca = nn.ModuleDict(
            {str(i): m for i, m in lacca_por_etapa.items()})

    def forward(self, x):
        feats = list(self.enc(x))
        for clave, mod in self.lacca.items():
            i = int(clave)
            feats[i] = mod(feats[i])
        return feats

    def __getattr__(self, nombre):
        try:
            return super().__getattr__(nombre)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["enc"], nombre)


class UNetLACCA(nn.Module):
    """U-Net (smp, resnet34, sin preentrenar) con LACCA en las etapas dadas.

    etapas: índices de features del encoder (resnet34: 3 -> 1/8 con 128 canales,
    4 -> 1/16 con 256 canales). Sin LACCA (etapas=()) es la U-Net base.
    Salida: 1 canal logit, igual que el resto del benchmark.
    """

    def __init__(self, encoder_name="resnet34", etapas=(3, 4), in_channels=1,
                 classes=1, **kw_lacca):
        super().__init__()
        import segmentation_models_pytorch as smp
        base = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                        in_channels=in_channels, classes=classes)
        canales = base.encoder.out_channels
        mods = {i: LACCA(canales[i], **kw_lacca) for i in etapas}
        base.encoder = _EncoderConLACCA(base.encoder, mods)
        self.base = base
        self.etapas = tuple(etapas)

    def forward(self, x):
        return self.base(x)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.sigmoid(self.forward(x))


ModeloOriginal = UNetLACCA


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
