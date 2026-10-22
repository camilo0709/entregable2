"""
figuras_benchmark.py — figuras y cifras de las Secciones 6 y 7.

Entradas (selladas o regenerables):
    resultados/resumen_runs.csv      (50 runs completados)
    resultados/perfil_modelos.csv    (params, GMACs, latencia CPU)

Salidas:
    libro/book/figs_stats/dice_ci.png     medias de Dice_val con IC95 bootstrap
    libro/book/figs_stats/cd_nemenyi.png  diagrama de diferencia crítica
    libro/book/figs_stats/pareto.png      frontera de Pareto (Dice vs coste)
    libro/book/figs_stats/ruta_i1_i8.png  diagrama de ruta para imágenes

Uso:
    python codigo/figuras_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "codigo"))

RES = os.path.join(ROOT, "resultados")
OUT = os.path.join(ROOT, "libro", "book", "figs_stats")
os.makedirs(OUT, exist_ok=True)

SEMILLAS = [42, 123, 777, 2024, 31415]
MODELOS = ["unet", "unet_pp", "deeplabv3p", "transunet",
           "swin_unet", "unetr", "segformer", "mednext"]
ETIQUETAS = {
    "unet": "U-Net", "unet_pp": "U-Net++", "deeplabv3p": "DeepLabV3+",
    "transunet": "TransUNet", "swin_unet": "Swin-UNet", "unetr": "UNETR",
    "segformer": "SegFormer", "mednext": "MedNeXt",
}
COLOR = "#2f5d8a"


# ------------------------------------------------------------------ datos
def cargar(dir_res):
    filas = [r for r in csv.DictReader(open(os.path.join(dir_res, "resumen_runs.csv"),
                                            encoding="utf-8"))
             if r["estado"] == "completado"]
    dice = {(r["modelo"], int(r["semilla"])): float(r["dice_val"]) for r in filas}
    seg = {(r["modelo"], int(r["semilla"])): float(r["segundos"]) for r in filas}
    ruta_perfil = os.path.join(dir_res, "perfil_modelos.csv")
    if not os.path.exists(ruta_perfil):
        ruta_perfil = os.path.join(RES, "perfil_modelos.csv")
    perfil = {r["modelo"]: r for r in csv.DictReader(
        open(ruta_perfil, encoding="utf-8"))}
    return dice, seg, perfil


def resumen(dice):
    M = {m: np.array([dice[(m, s)] for s in SEMILLAS]) for m in MODELOS}
    out = {}
    rng = np.random.default_rng(42)
    for m in MODELOS:
        v = M[m]
        bs = v[rng.integers(0, 5, (20000, 5))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[m] = {"media": float(v.mean()), "sd": float(v.std(ddof=1)),
                  "lo": float(lo), "hi": float(hi),
                  "cv": float(100 * v.std(ddof=1) / v.mean())}
    return M, out


def rangos(dice):
    R = np.zeros((len(SEMILLAS), len(MODELOS)))
    for j, s in enumerate(SEMILLAS):
        R[j] = stats.rankdata([-dice[(m, s)] for m in MODELOS])
    return {m: float(R[:, i].mean()) for i, m in enumerate(MODELOS)}


def cd_nemenyi(k=8, n=5, alpha=0.05):
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n))), float(q)


# --------------------------------------------------------------- figura 1
def fig_dice_ci(res):
    orden = sorted(MODELOS, key=lambda m: res[m]["media"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(orden))
    vals = [res[m]["media"] for m in orden]
    lo = [res[m]["media"] - res[m]["lo"] for m in orden]
    hi = [res[m]["hi"] - res[m]["media"] for m in orden]
    ax.barh(y, vals, color=COLOR, alpha=0.85, height=0.6)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor="black",
                capsize=3, lw=1.2)
    ax.set_yticks(y, [ETIQUETAS[m] for m in orden])
    ax.set_xlim(0.42, 0.53)
    ax.set_xlabel("Dice de validación (media de 5 semillas; IC 95 % bootstrap)")
    ax.axvline(0.156, color="#b03a2e", ls="--", lw=1)
    ax.text(0.158, len(orden) - 0.35, "trivial todo-foreground 0,156",
            color="#b03a2e", fontsize=8, va="top")
    for yi, m in zip(y, orden):
        ax.text(res[m]["media"] + 0.001, yi, f"{res[m]['media']:.4f}",
                va="center", fontsize=8)
    ax.set_title("Dice de validación por modelo (8 modelos con runs válidos)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    ruta = os.path.join(OUT, "dice_ci.png")
    fig.savefig(ruta, dpi=190)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------- figura 2
def fig_cd(ranks, cd):
    orden = sorted(ranks.items(), key=lambda kv: kv[1])
    k = len(orden)
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.set_xlim(0.3, k + 0.7)
    ax.set_ylim(-0.5 * ((k + 1) // 2) - 1.0, 1.55)
    ax.axis("off")

    y0 = 0.0
    ax.plot([1, k], [y0, y0], color="black", lw=1.2)
    for r in range(1, k + 1):
        ax.plot([r, r], [y0, y0 + 0.07], color="black", lw=1)
        ax.text(r, y0 + 0.10, str(r), ha="center", va="bottom", fontsize=9)

    ax.plot([1, 1 + cd], [y0 + 0.55, y0 + 0.55], color="black", lw=2.2)
    for x in (1, 1 + cd):
        ax.plot([x, x], [y0 + 0.48, y0 + 0.62], color="black", lw=1.2)
    ax.text(1 + cd / 2, y0 + 0.68, f"CD = {cd:.2f}", ha="center", fontsize=9)
    ax.text(k, y0 + 0.10, "rango medio (1 = mejor)", ha="right",
            va="bottom", fontsize=8, color="gray")

    # etiquetas alternadas a izquierda/derecha
    for i, (m, r) in enumerate(orden):
        nivel = i // 2
        yy = -0.55 - 0.52 * nivel
        if i % 2 == 0:
            ax.text(0.45, yy, ETIQUETAS[m], ha="left", va="center", fontsize=9)
            ax.plot([0.9, r], [yy, y0 - 0.02], color="gray", lw=0.8)
        else:
            ax.text(k + 0.55, yy, ETIQUETAS[m], ha="right", va="center", fontsize=9)
            ax.plot([k + 0.1, r], [yy, y0 - 0.02], color="gray", lw=0.8)

    # grupos no significativos (cliques maximales)
    vals = [r for _, r in orden]
    grupos = []
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i:
            grupos.append((i, j))
    maximales = [g for g in grupos
                 if not any(h[0] <= g[0] and g[1] <= h[1] and h != g
                            for h in grupos)]
    for nivel, (i, j) in enumerate(maximales):
        yy = y0 - 0.20 - 0.11 * nivel
        ax.plot([vals[i], vals[j]], [yy, yy], color="black", lw=5,
                solid_capstyle="butt", alpha=0.55)

    ax.set_title("Diagrama de diferencia crítica (Nemenyi, 8 modelos x 5 semillas)")
    fig.tight_layout()
    ruta = os.path.join(OUT, "cd_nemenyi.png")
    fig.savefig(ruta, dpi=190)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------- figura 3
def frontera(puntos):
    """Puntos (coste, dice): maximizar dice, minimizar coste."""
    orden = sorted(puntos, key=lambda x: (x[1], -x[2]))
    frente, mejor = [], -np.inf
    for m, c, d in orden:
        if c == 0:
            continue
        if d > mejor:
            frente.append((m, c, d))
            mejor = d
    return frente


def fig_pareto(res, perfil):
    costes = {
        "gmacs": ("GMACs (512x512)", lambda m: float(perfil[m]["gmacs"])),
        "latencia": ("latencia CPU (ms)", lambda m: float(perfil[m]["latencia_ms_mediana"])),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for ax, (clave, (etiqueta, fn)) in zip(axes, costes.items()):
        pts = [(m, fn(m), res[m]["media"]) for m in MODELOS]
        fr = frontera(pts)
        fm = {p[0] for p in fr}
        for m, c, d in pts:
            ax.scatter(c, d, s=55, color=COLOR if m in fm else "lightgray",
                       edgecolor="black", lw=0.7, zorder=3)
            ax.annotate(ETIQUETAS[m], (c, d), textcoords="offset points",
                        xytext=(5, 5), fontsize=7.5)
        fr = sorted(fr, key=lambda x: x[1])
        ax.plot([p[1] for p in fr], [p[2] for p in fr], color="#b03a2e",
                lw=1.4, ls="--", zorder=2, label="frontera de Pareto")
        ax.set_xlabel(etiqueta)
        ax.set_ylabel("Dice de validación")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Frontera de Pareto: calidad frente a coste computacional")
    fig.tight_layout()
    ruta = os.path.join(OUT, "pareto.png")
    fig.savefig(ruta, dpi=190)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------- figura 4
def fig_ruta():
    pasos = [
        ("I1. Tarea y unidad", "Segmentación binaria;\nunidad = imagen\n(sin ID de paciente)"),
        ("I2. Duplicados y partición", "MD5 + pHash (840 imgs);\nsplits sellados\n165 / 29 / 80 / 40"),
        ("I3. Normalización", "512x512, 1 canal;\nz-score por imagen\n(sin estadísticos globales)"),
        ("I4. Aumentos", "flip H/V, brillo y ruido;\nsolo train e imagen+máscara\nsincronizadas"),
        ("I5. Modelos y baselines", "10 modelos + máscara vacía,\nbbox y unet_fuerte;\nmismo presupuesto"),
        ("I6. Inferencia", "umbral 0,5 fijo;\nsin TTA;\nmisma política para todos"),
        ("I7. Pruebas", "Friedman + Nemenyi;\nWilcoxon-Holm; bootstrap;\nδ = 0,02"),
        ("I8. Interpretabilidad", "Grad-CAM vs caja/máscara;\nsubgrupos; tasa de FP\nen test_neg"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ancho, alto = 4.15, 1.0
    xs = [0.45, 5.35]
    ys = [4.75, 3.35, 1.95, 0.55]
    cajas = []
    for i, (titulo, texto) in enumerate(pasos):
        col, fila = i % 2, i // 2
        x, y = xs[col], ys[fila]
        caja = FancyBboxPatch((x, y), ancho, alto,
                              boxstyle="round,pad=0.04,rounding_size=0.08",
                              linewidth=1.1, edgecolor=COLOR,
                              facecolor="#eef3f8")
        ax.add_patch(caja)
        ax.text(x + 0.15, y + alto - 0.22, titulo, fontsize=9,
                fontweight="bold", color="#1d3a57", va="top")
        ax.text(x + 0.15, y + alto - 0.50, texto, fontsize=7.8,
                va="top", color="#222222")
        cajas.append((x, y))

    def flecha(p1, p2):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                     color=COLOR, lw=1.2,
                                     connectionstyle="arc3,rad=0"))

    for i in range(len(pasos) - 1):
        x1, y1 = cajas[i]
        x2, y2 = cajas[i + 1]
        if i % 2 == 0:  # mismo renglón, hacia la derecha
            flecha((x1 + ancho, y1 + alto / 2), (x2, y2 + alto / 2))
        else:           # bajar al siguiente renglón
            flecha((x1 + ancho / 2, y1), (x2 + ancho / 2, y2 + alto))

    ax.text(5, 5.75, "Ruta para datos de imágenes (TBX11K): I1-I8",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(5, 0.08, "Salida: Tablas A/B/C, frontera de Pareto y Grad-CAM",
            ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    ruta = os.path.join(OUT, "ruta_i1_i8.png")
    fig.savefig(ruta, dpi=190)
    plt.close(fig)
    return ruta


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default=RES,
                    help="carpeta con resumen_runs.csv (por defecto, resultados/)")
    args = ap.parse_args()
    dice, seg, perfil = cargar(args.resultados)
    M, res = resumen(dice)
    ranks = rangos(dice)
    cd, q = cd_nemenyi()
    chi2, p = stats.friedmanchisquare(*[M[m] for m in MODELOS])
    fid = ((len(SEMILLAS) - 1) * chi2) / (len(SEMILLAS) * (len(MODELOS) - 1) - chi2)
    p_id = stats.f.sf(fid, len(MODELOS) - 1,
                      (len(MODELOS) - 1) * (len(SEMILLAS) - 1))
    print(f"Friedman chi2={chi2:.3f} p={p:.6f}")
    print(f"Iman-Davenport F={fid:.3f} p={p_id:.3e}")
    print(f"Nemenyi q={q:.3f} CD={cd:.3f}")
    for m in MODELOS:
        print(f"{m:12s} media={res[m]['media']:.4f} sd={res[m]['sd']:.4f} "
              f"IC=[{res[m]['lo']:.4f},{res[m]['hi']:.4f}] "
              f"cv={res[m]['cv']:.1f}% rango={ranks[m]:.2f} "
              f"t={np.mean([seg[(m, s)] for s in SEMILLAS])/60:.1f} min")
    for ruta in (fig_dice_ci(res), fig_cd(ranks, cd), fig_pareto(res, perfil),
                 fig_ruta()):
        print("figura:", os.path.relpath(ruta, ROOT))


if __name__ == "__main__":
    main()
