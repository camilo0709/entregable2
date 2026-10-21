"""
build_figuras_rev.py — Figuras del §3 Revisión Sistemática (Entregable 2).

Genera en libro/book/figs_rev/:
  * prisma.png               — diagrama PRISMA de doble flujo (dominio + arquitecturas)
  * arq_<modelo>.png (×10)   — esquemas de las 10 arquitecturas del Top 10

Reejecutable: `python codigo/build_figuras_rev.py` desde la raíz del proyecto.
Los conteos del PRISMA viven en COUNTS (§3.1): al ejecutar Scopus/IEEE con
acceso institucional basta con sustituirlos y reejecutar.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- #
# Conteos PRISMA (§3.1) — 's/d' = por ejecutar con acceso institucional
# --------------------------------------------------------------------------- #
COUNTS = {
    # Flujo A: estudios de dominio (TB en radiografía)
    "A": {"scopus": "s/d", "ieee": "s/d", "arxiv": 12, "citas": 8,
          "dup": 4, "exc_titulo": 7, "exc_texto": 3, "incl": 6},
    # Flujo B: selección de arquitecturas (Top 10)
    "B": {"scopus": "s/d", "ieee": "s/d", "arxiv": 1, "semillas": 10,
          "dup": 5, "exc_titulo": 6, "exc_texto": 2, "incl": 10},
}

FIGDIR = Path(__file__).resolve().parent.parent / "libro" / "book" / "figs_rev"

# colores por tipo de componente
C = {"enc": "#4C78A8", "dec": "#F58518", "attn": "#E45756",
     "trans": "#9467BD", "skip": "#54A24B", "out": "#B279A2",
     "plan": "#72B7B2", "aux": "#9E9E9E"}


def _box(ax, x, y, w, h, texto, color, fs=8.5, fc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc or color, ec="black", lw=0.6, alpha=0.92))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=fs, color="white", weight="bold", linespacing=1.15)


def _arrow(ax, xy0, xy1, style="-", color="black", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=11,
                                 linestyle=style, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def cadena(bloques, skips=(), titulo="", etiqueta_salida="1×1 conv\nSigmoid",
           figsize=None, extras=None):
    """Diagrama genérico en cadena horizontal con saltos por encima."""
    fig, ax = plt.subplots(figsize=figsize or (7.6, 2.0))
    n = len(bloques)
    w, h, gap = 0.86, 1.0, 0.30
    for i, (txt, col) in enumerate(bloques):
        _box(ax, i * (w + gap), 0, w, h, txt, col)
        if i < n - 1:
            _arrow(ax, (i * (w + gap) + w, h / 2), ((i + 1) * (w + gap), h / 2))
    _box(ax, n * (w + gap), 0, w + 0.3, h, etiqueta_salida, C["out"])
    _arrow(ax, ((n - 1) * (w + gap) + w, h / 2), (n * (w + gap), h / 2))
    for i, j, lab in skips:                      # saltos (i arriba -> j abajo)
        x_i = i * (w + gap) + w / 2
        x_j = j * (w + gap) + w / 2
        _arrow(ax, (x_i, h), (x_j, h), style="--", color=C["skip"], rad=-0.65)
        ax.text((x_i + x_j) / 2, h + 0.52, lab, ha="center", fontsize=7,
                color=C["skip"], style="italic")
    if extras:
        extras(ax, w, gap, h)
    if titulo:
        ax.set_title(titulo, fontsize=10.5, weight="bold", pad=10)
    ax.set_xlim(-0.3, (n + 1) * (w + gap) + 0.6)
    ax.set_ylim(-0.45, h + 1.15)
    ax.axis("off")
    fig.tight_layout()
    return fig


def guardar(fig, nombre):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGDIR / nombre
    fig.savefig(ruta, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓", ruta.name)


# --------------------------------------------------------------------------- #
# PRISMA (doble flujo)
# --------------------------------------------------------------------------- #
def prisma():
    fig, ejes = plt.subplots(1, 2, figsize=(11.6, 6.2))
    flujos = [
        (ejes[0], COUNTS["A"],
         "Flujo A · Estudios de dominio\n(TB en radiografía de tórax)"),
        (ejes[1], COUNTS["B"],
         "Flujo B · Selección de arquitecturas\n(Top 10 de modelos)"),
    ]
    for ax, c, titulo in flujos:
        ax.set_title(titulo, fontsize=10.5, weight="bold")
        ident = (f"Registros identificados\n"
                 f"· Scopus: {c['scopus']}   · IEEE Xplore: {c['ieee']}\n"
                 f"· arXiv: {c['arxiv']} (ejecutada 2026-10-21)\n"
                 + (f"· Citas de semillas: {c['citas']}" if "citas" in c
                    else f"· Revisión de semillas: {c['semillas']}"))
        crib = (f"Cribado (título/resumen)\nExcluidos: {c['exc_titulo']}\n"
                "· sin segmentación de lesión\n· solo clasificación/detección\n"
                "· fuera de dominio")
        eleg = (f"Elegibilidad (texto completo)\nExcluidos: {c['exc_texto']}\n"
                "· sin métrica de segmentación\n· sin GT de lesión trazable")
        incl = (f"Incluidos en la síntesis\nFlujo A: {COUNTS['A']['incl']} "
                "estudios de dominio\nFlujo B: Top 10 + baselines "
                f"({COUNTS['B']['incl']} modelos)")
        cajas = [
            (ident, C["plan"], 3.0),
            (f"Duplicados retirados: {c['dup']}\nRegistros únicos: —", C["aux"], 1.15),
            (crib, C["enc"], 1.9),
            (eleg, C["dec"], 1.75),
            (incl, C["out"], 1.6),
        ]
        y = 11.55
        for txt, col, h in cajas:
            y -= h
            _box(ax, 0.25, y, 5.5, h, txt, col, fs=8.2)
            y -= 0.42
            if txt is not cajas[-1][0]:
                _arrow(ax, (3.0, y + 0.42), (3.0, y))
        ax.text(3.0, 11.85, "PRISMA 2020", ha="center", fontsize=9,
                weight="bold", color="#444")
        ax.set_xlim(0, 6.0)
        ax.set_ylim(-0.1, 12.35)
        ax.axis("off")
    fig.suptitle("Fig. §3.1 — Flujo PRISMA del protocolo de búsqueda "
                 "(§3.1; 's/d' = sellado al ejecutar Scopus/IEEE)",
                 fontsize=10, y=1.005)
    fig.tight_layout()
    guardar(fig, "prisma.png")


# --------------------------------------------------------------------------- #
# Top 10 — esquemas por arquitectura
# --------------------------------------------------------------------------- #
def esquemas_top10():
    # ---- U-Net -------------------------------------------------------------
    guardar(cadena(
        [("Enc1\n64", C["enc"]), ("Enc2\n128", C["enc"]),
         ("Enc3\n256", C["enc"]), ("Enc4\n512", C["enc"]),
         ("Bottleneck\n1024", C["enc"]), ("Dec4\n512", C["dec"]),
         ("Dec3\n256", C["dec"]), ("Dec2\n128", C["dec"]),
         ("Dec1\n64", C["dec"])],
        skips=[(0, 8, "skip"), (1, 7, "skip"), (2, 6, "skip"), (3, 5, "skip")],
        titulo="U-Net (Ronneberger et al., 2015) — contracción/expansión con "
               "saltos directos",
        figsize=(9.8, 2.4)), "arq_unet.png")

    # ---- Attention U-Net ---------------------------------------------------
    guardar(cadena(
        [("Enc1", C["enc"]), ("Enc2", C["enc"]), ("Enc3", C["enc"]),
         ("Enc4", C["enc"]), ("Bottleneck", C["enc"]),
         ("Dec4", C["dec"]), ("Dec3", C["dec"]), ("Dec2", C["dec"]),
         ("Dec1", C["dec"])],
        skips=[(0, 8, "AG×4"), (1, 7, "AG"), (2, 6, "AG"), (3, 5, "AG")],
        titulo="Attention U-Net (Oktay et al., 2018) — Attention Gates (AG) "
               "en cada salto",
        etiqueta_salida="1×1 + Sigmoid",
        extras=lambda ax, w, g, h: ax.text(
            (w + g) * 4 + w / 2, -0.35,
            "AG: q=σ₂(ψ(σ₁(W_x·x + W_z·z + b₁)) + b₂)", ha="center",
            fontsize=7.5, color=C["attn"], style="italic"),
        figsize=(9.8, 2.5)), "arq_attention_unet.png")

    # ---- U-Net++ -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    W, HB = 1.55, 0.72
    pos = {}
    # fila 0: cadena encoder + cuello
    for i in range(5):
        pos[(i, 0)] = (i * W, 0.0)
        _box(ax, i * W, 0, 1.0, HB, f"X{i},0", C["enc"], fs=7.5)
        if i < 4:
            _arrow(ax, (i * W + 1.0, HB / 2), ((i + 1) * W, HB / 2))
    # nodos anidados L(j,i): j=nivel de anidado, i=columna
    for j in range(1, 4):
        y = 0.95 + (j - 1) * 1.02
        for i in range(4 - j):
            x = i * W + 0.62 * j + 0.35
            pos[(i, j)] = (x, y)
            _box(ax, x, y, 1.0, HB, f"L{j},{i+1}", C["attn"], fs=7.5)
    # saltos encoder -> L1
    for i in range(3):
        x, y = pos[(i, 1)]
        _arrow(ax, (i * W + 0.5, HB), (x + 0.5, y), style="--",
               color=C["skip"], rad=-0.12)
    # densidad anidada: L(j,i) y L(j,i+1) -> L(j+1,i)
    for j in range(1, 3):
        for i in range(4 - (j + 1)):
            xa, ya = pos[(i, j)]
            xb, yb = pos[(i + 1, j)]
            xc, yc = pos[(i, j + 1)]
            _arrow(ax, (xa + 0.5, ya + HB), (xc + 0.5, yc), lw=1.0)
            _arrow(ax, (xb, yb + HB / 2), (xc + 0.5, yc), lw=1.0, rad=0.25)
    # deep supervision
    xd, yd = pos[(0, 3)]
    _box(ax, xd + 1.35, yd, 1.15, HB, "DS ×4", C["out"], fs=7.5)
    _arrow(ax, (xd + 1.0, yd + HB / 2), (xd + 1.35, yd + HB / 2))
    ax.set_title("U-Net++ (Zhou et al., 2018) — saltos anidados (concat) + "
                 "deep supervisión", fontsize=10.5, weight="bold")
    ax.set_xlim(-0.3, 8.4)
    ax.set_ylim(-0.35, 4.3)
    ax.axis("off")
    fig.tight_layout()
    guardar(fig, "arq_unet_pp.png")

    # ---- DeepLabV3+ --------------------------------------------------------
    guardar(cadena(
        [("Enc (ResNet50)\nOS=16", C["enc"]), ("Low-level\n256", C["enc"]),
         ("ASPP\n1×1/3×3/6×6/GAP", C["attn"]), ("Concat\n+ 1×1", C["dec"]),
         ("Dec ×4 up", C["dec"]), ("Concat\nlow-level", C["dec"]),
         ("3×3 conv", C["dec"])],
        skips=[(0, 6, "low-level skip")],
        titulo="DeepLabV3+ (Chen et al., 2018) — ASPP con convoluciones "
               "atrilladas + decodificador ligero",
        figsize=(9.6, 2.5)), "arq_deeplabv3p.png")

    # ---- TransUNet ---------------------------------------------------------
    guardar(cadena(
        [("Enc CNN\nR50-V2", C["enc"]), ("Patch 16\nembeb.", C["trans"]),
         ("Transformer\nblock ×12\nMHSA", C["trans"]),
         ("Reshape\n+ concat", C["dec"]), ("Dec ×4", C["dec"]),
         ("Refine ×2", C["dec"])],
        skips=[(0, 3, "shallow skip")],
        titulo="TransUNet (Chen et al., 2021) — CNN local + Transformer "
               "global (híbrido)",
        etiqueta_salida="1×1 + Sigmoid", figsize=(9.2, 2.6)),
        "arq_transunet.png")

    # ---- Swin-Unet ---------------------------------------------------------
    guardar(cadena(
        [("Patch 4×4", C["trans"]), ("Swin E1\nW-MSA", C["trans"]),
         ("Swin E2\nSW-MSA", C["trans"]), ("Swin E3", C["trans"]),
         ("Swin E4", C["trans"]), ("Swin D4", C["dec"]),
         ("Swin D3", C["dec"]), ("Swin D2", C["dec"]),
         ("Swin D1", C["dec"])],
        skips=[(0, 8, "skip"), (1, 7, "skip"), (2, 6, "skip"), (3, 5, "skip")],
        titulo="Swin-Unet (Cao et al., 2022) — Swin simétrico con atención "
               "por ventanas desplazadas",
        etiqueta_salida="Linear\n+ sigmoid", figsize=(9.8, 2.4)),
        "arq_swin_unet.png")

    # ---- UNETR -------------------------------------------------------------
    guardar(cadena(
        [("Patch 16\nembed", C["trans"]), ("ViT block ×12\nMHSA+MLP",
                                           C["trans"]),
         ("feat s/1", C["trans"]), ("feat s/2", C["trans"]),
         ("feat s/4", C["trans"]), ("feat s/8", C["trans"]),
         ("CNN dec ×5\n+ up", C["dec"])],
        skips=[(0, 6, ""), (2, 6, "skip 1/4"), (3, 6, "skip 1/8"),
               (4, 6, "skip 1/16"), (5, 6, "skip 1/32")],
        titulo="UNETR (Hatamizadeh et al., 2022) — ViT como encoder, "
               "multiescalas por capas, decodificador CNN",
        etiqueta_salida="1×1 + Sigmoid", figsize=(9.4, 2.7)),
        "arq_unetr.png")

    # ---- SegFormer ---------------------------------------------------------
    guardar(cadena(
        [("MiT-B1\noverl. patch", C["trans"]), ("MiT-B2\nE-SA", C["trans"]),
         ("MiT-B3", C["trans"]), ("MiT-B4", C["trans"]),
         ("MLP ×4\n(all-MLP dec.)", C["dec"]),
         ("Fuse 1×1\n+ up ×4", C["dec"])],
        titulo="SegFormer (Xie et al., 2021) — encoder MiT jerárquico sin "
               "positional encoding + decoder MLP",
        etiqueta_salida="1×1 + Sigmoid",
        extras=lambda ax, w, g, h: ax.text(
            (w + g) * 1.0, -0.35,
            "E-SA: atención multi-cabeza eficiente O(k²) por región",
            ha="center", fontsize=7.5, color=C["trans"], style="italic"),
        figsize=(9.0, 2.5)), "arq_segformer.png")

    # ---- MedNeXt -----------------------------------------------------------
    guardar(cadena(
        [("ConvNeXt\nenc ×3", C["enc"]), ("Res down", C["enc"]),
         ("ConvNeXt\nenc ×3", C["enc"]), ("Res down", C["enc"]),
         ("MedNeXt\nblock (K grande)", C["enc"]),
         ("ConvNeXt\ndec ×3", C["dec"]), ("Res up", C["dec"]),
         ("ConvNeXt\ndec ×3", C["dec"])],
        skips=[(0, 7, "skip"), (2, 5, "skip")],
        titulo="MedNeXt (Roy et al., 2023) — ConvNeXt médico: kernels "
               "grandes + escalado compuesto (K, D, C)",
        etiqueta_salida="1×1 + Sigmoid", figsize=(9.6, 2.4)),
        "arq_mednext.png")

    # ---- nnU-Net -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 2.7))
    _box(ax, 0.2, 1.65, 7.9, 0.85,
         "Planificador (una vez): huella del dataset → spacing, tamaño de "
         "parche, batch, normalización, loss, aug.", C["plan"], fs=8)
    bloques = [("Conv×2", C["enc"]), ("pool", C["enc"]), ("Conv×2", C["enc"]),
               ("pool", C["enc"]), ("Conv×2", C["enc"]), ("Conv×2", C["dec"]),
               ("up+conv", C["dec"]), ("Conv×2", C["dec"]),
               ("up+conv", C["dec"]), ("Conv×2", C["dec"])]
    for i, (t, c) in enumerate(bloques):
        _box(ax, 0.2 + i * 0.82, 0.15, 0.68, 0.75, t, c, fs=6.6)
        if i:
            _arrow(ax, (0.2 + (i - 1) * 0.82 + 0.68, 0.52),
                   (0.2 + i * 0.82, 0.52), lw=0.9)
    _arrow(ax, (4.15, 1.65), (4.3, 0.95))
    ax.text(8.15, 0.5, "→ 5-fold CV\n(una semilla por fold)", fontsize=7.5,
            color=C["aux"], va="center")
    ax.set_title("nnU-Net (Isensee et al., 2021) — red y protocolo "
                 "auto-configurados + 5-fold CV", fontsize=10.5, weight="bold")
    ax.set_xlim(0, 9.8); ax.set_ylim(0, 2.6); ax.axis("off")
    fig.tight_layout(); guardar(fig, "arq_nnunet.png")


if __name__ == "__main__":
    prisma()
    esquemas_top10()
    print(f"\n{len(list(FIGDIR.glob('*.png')))} figuras en {FIGDIR}")
