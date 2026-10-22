"""
estadistica_tb.py — inferencia estadística del benchmark (Secciones 6 y 7).

Entrada (generada por el driver y por evaluar_test.py):
    resultados/<modelo>/seed_<semilla>/resultado.json      (dice_val, dice_test)
    resultados/<modelo>/seed_<semilla>/por_imagen.csv      (val, por imagen)
    resultados/<modelo>/seed_<semilla>/por_imagen_test.csv (test, por imagen)
    resultados/perfil_modelos.csv                          (params/FLOPs/latencia)

Salida: resultados/estadistica/
    matriz_test.csv        matriz modelo x semilla (verificación del diseño)
    tabla_A_runs.csv       corrida a corrida (dice_val, dice_test, HD95, ASSD...)
    tabla_B_inferencia.csv resumen por modelo + Friedman/Nemenyi/Wilcoxon-Holm
                           + IC95 bootstrap + veredicto contra delta
    tabla_C_coste.csv      calidad vs coste (params, GMACs, latencia, tiempo)
    inferencia.json        todo lo anterior en forma machine-readable

Diseño de inferencia (preregistro Seccion 1.1, δ = 0.02):
  * Unidad experimental: la corrida (modelo x semilla); n = 5 semillas.
  * Metrica primaria: Dice medio por imagen en TEST (GT plata D1-R1,
    umbral fijo 0.5). La metrica de validacion (dice_val) se reporta como
    diagnostico, nunca como resultado principal.
  * Omnibus: Friedman por bloques (semillas) + correccion de Iman-Davenport.
  * Post-hoc: Nemenyi (CD, q del rango estudiantilizado) y Wilcoxon con
    signo EXACTO por par contra el modelo de referencia (mayor media),
    con desplazamiento delta (doble significancia) y correccion de Holm.
  * IC 95% por bootstrap jerarquico (remuestreo de semillas y de imagenes
    dentro de semilla; B = 10000) de la media de cada modelo y de la
    diferencia contra la referencia.
  * Con n = 5 el minimo p exacto bilateral es 0.0625: el test exacto no
    puede rechazar nada a 0.05 tras Holm. Por eso el veredicto formal se
    apoya en el IC bootstrap del desplazado delta (se reportan ambos).
  * Verificacion de duplicados: compara las curvas de loss entre modelos
    en la misma semilla; curvas identicas = misma red entrenada dos veces
    (se excluyen y se avisa). Surgio de un bug real (attention_type
    ignorado por smp) y se mantiene como red de seguridad.

Uso:
    python codigo/estadistica_tb.py --resultados resultados
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driver_entrenamiento import ruta_run, run_completado   # noqa: E402
from modelos_tb import (MODELOS, ORDEN_BASELINES,           # noqa: E402
                        ORDEN_BENCHMARK, SEMILLAS)

DELTA = 0.02          # δ del preregistro (equivalencia práctica)
ALPHA = 0.05
B_BOOT = 10000        # iteraciones de bootstrap


# ------------------------------------------------------------------ carga
def cargar_runs(res_dir, modelos, semillas):
    """Devuelve datos[(modelo, semilla)] = dict de resultado.json."""
    datos = {}
    for m in modelos:
        for s in semillas:
            if not run_completado(res_dir, m, s):
                continue
            ruta = os.path.join(ruta_run(res_dir, m, s), "resultado.json")
            with open(ruta, encoding="utf-8") as f:
                datos[(m, s)] = json.load(f)
    return datos


def cargar_imagenes(res_dir, m, s, cual="test"):
    """Vector de Dice por imagen (test o val) de una corrida."""
    nombre = "por_imagen_test.csv" if cual == "test" else "por_imagen.csv"
    ruta = os.path.join(ruta_run(res_dir, m, s), nombre)
    if not os.path.exists(ruta):
        return None
    df = pd.read_csv(ruta)
    return df


# ------------------------------------------------------- bootstrap básico
def ic_bootstrap_semillas(valores_semillas, rng, b=B_BOOT):
    """IC 95% de la media: remuestrea SEMILLAS (n corridas independientes)."""
    v = np.asarray(valores_semillas, dtype=float)
    idx = rng.integers(0, len(v), size=(b, len(v)))
    medias = v[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return float(lo), float(hi)


def ic_bootstrap_jerarquico(matriz, rng, b=5000):
    """IC 95% jerarquico: remuestrea semillas y luego imagenes por semilla.

    matriz: lista de arrays (uno por semilla) con el Dice por imagen.
    """
    n = len(matriz)
    medias = np.empty(b)
    for k in range(b):
        ss = rng.integers(0, n, size=n)              # semillas
        valores = []
        for si in ss:
            v = matriz[si]
            valores.append(v[rng.integers(0, len(v), size=len(v))].mean())
        medias[k] = np.mean(valores)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return float(lo), float(hi), medias


def dif_jerarquica(matriz_ref, matriz_m, rng, b=5000):
    """IC 95% de (ref - modelo) con remuestreo PAREADO por semilla."""
    n = len(matriz_ref)
    medias = np.empty(b)
    for k in range(b):
        ss = rng.integers(0, n, size=n)
        difs = []
        for si in ss:
            vr, vm = matriz_ref[si], matriz_m[si]
            ir = rng.integers(0, len(vr), size=len(vr))
            im = rng.integers(0, len(vm), size=len(vm))
            difs.append(vr[ir].mean() - vm[im].mean())
        medias[k] = np.mean(difs)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvalores):
    """Correccion de Holm (step-down) sobre una lista de p."""
    m = len(pvalores)
    orden = np.argsort(pvalores)
    ajustados = np.empty(m)
    acumulado = 0.0
    for pos, i in enumerate(orden):
        acumulado = max(acumulado, (m - pos) * pvalores[i])
        ajustados[i] = min(acumulado, 1.0)
    return ajustados


def verificar_duplicados(datos, modelos, semillas):
    """Curvas de loss identicas entre modelos en la misma semilla = bug."""
    sospechosos = []
    for s in semillas:
        curvas = {}
        for m in modelos:
            r = datos.get((m, s))
            if r and r.get("historial"):
                curvas[m] = tuple(round(h["loss"], 9)
                                  for h in r["historial"])
        vistos = {}
        for m, c in curvas.items():
            if c in vistos:
                sospechosos.append({"semilla": s, "modelos": [vistos[c], m]})
            else:
                vistos[c] = m
    return sospechosos


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Inferencia estadistica (S6/S7)")
    ap.add_argument("--resultados", default=None)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--semillas", nargs="*", type=int, default=SEMILLAS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args()

    raiz = Path(__file__).resolve().parents[1]
    res_dir = a.resultados or os.path.join(raiz, "resultados")
    out = os.path.join(res_dir, "estadistica")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261022)

    entrenables = [m for m in ORDEN_BENCHMARK + ORDEN_BASELINES
                   if MODELOS.get(m, {}).get("entrenable", False)]
    datos = cargar_runs(res_dir, entrenables, a.semillas)

    # ------------------------------------------------ verificación previa
    duplicados = verificar_duplicados(datos, entrenables, a.semillas)
    print("=" * 74)
    print("ESTADISTICA DEL BENCHMARK")
    print("=" * 74)
    print(f"Corridas completadas encontradas: {len(datos)}")
    if duplicados:
        print("¡¡ ALERTA: curvas identicas entre modelos (posible bug)!!")
        for d in duplicados:
            print(f"   semilla {d['semilla']}: {d['modelos']}")
    else:
        print("Verificacion de duplicados: sin curvas identicas entre "
              "modelos distintos.")

    # ------------------------------------------------- matriz y elegibles
    matriz_test, elegidos, excluidos = {}, [], []
    for m in entrenables:
        filas = [datos[(m, s)] for s in a.semillas if (m, s) in datos]
        con_test = [r for r in filas if r.get("dice_test") is not None]
        if len(filas) == len(a.semillas) and len(con_test) == len(a.semillas):
            elegidos.append(m)
        else:
            excluidos.append({"modelo": m, "corridas": len(filas),
                              "con_dice_test": len(con_test),
                              "motivo": "matriz incompleta o sin evaluacion "
                                        "de test"})
    print(f"Modelos con matriz completa (5 semillas + test): {len(elegidos)}")
    for e in excluidos:
        print(f"   excluido: {e['modelo']} ({e['corridas']} corridas, "
              f"{e['con_dice_test']} con test)")

    # matriz modelo x semilla (para el informe)
    with open(os.path.join(out, "matriz_test.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo"] + [f"seed_{s}" for s in a.semillas])
        for m in elegidos:
            w.writerow([m] + [datos[(m, s)]["dice_test"] for s in a.semillas])

    if len(elegidos) < 3:
        print("Menos de 3 modelos completos: no hay inferencia posible.")
        return

    # --------------------------------------------------------- Tabla A
    tabla_A = []
    for m in elegidos:
        for s in a.semillas:
            r = datos[(m, s)]
            df = cargar_imagenes(res_dir, m, s, "test")
            tabla_A.append({
                "modelo": m, "semilla": s,
                "dice_val": r["dice_val"], "dice_test": r["dice_test"],
                "iou_test": round(float(df["iou"].mean()), 5),
                "hd95_test": (round(float(pd.to_numeric(
                    df["hd95"], errors="coerce").mean()), 3)
                    if "hd95" in df else None),
                "assd_test": (round(float(pd.to_numeric(
                    df["assd"], errors="coerce").mean()), 3)
                    if "assd" in df else None),
                "epoch_best": r.get("epoch_best"),
                "epochs_entrenadas": r.get("epochs_entrenadas"),
                "segundos": r.get("segundos"),
                "ood_frac_media": (r.get("ood") or {}).get("frac_media"),
            })
    pd.DataFrame(tabla_A).to_csv(os.path.join(out, "tabla_A_runs.csv"),
                                 index=False)

    # ------------------------------------------------------ omnibus (S6)
    muestras = [[datos[(m, s)]["dice_test"] for s in a.semillas]
                for m in elegidos]
    chi2, p_friedman = stats.friedmanchisquare(*muestras)
    n, k = len(a.semillas), len(elegidos)
    denominador = n * (k - 1) - chi2
    if denominador <= 0:
        # caso degenerado (separación perfecta): F = infinito, p = 0
        f_id, p_id = float("inf"), 0.0
    else:
        f_id = ((n - 1) * chi2) / denominador
        p_id = stats.f.sf(f_id, k - 1, (k - 1) * (n - 1))

    # rangos por semilla (1 = mejor dice_test)
    ranks = np.zeros((n, k))
    for j, s in enumerate(a.semillas):
        vals = [datos[(m, s)]["dice_test"] for m in elegidos]
        ranks[j] = stats.rankdata([-v for v in vals])   # mayor = rango 1
    rank_medio = ranks.mean(axis=0)

    # Nemenyi CD
    q = stats.studentized_range.ppf(1 - a.alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))

    # ------------------------------------------- modelos por media (ref)
    medias = np.array([np.mean(muestras[i]) for i in range(k)])
    desv = np.array([np.std(muestras[i], ddof=1) for i in range(k)])
    orden_ref = np.argsort(-medias)
    ref_i = int(orden_ref[0])
    ref = elegidos[ref_i]
    print(f"\nFriedman: chi2={chi2:.3f} p={p_friedman:.4g} | "
          f"Iman-Davenport F={f_id:.3f} p={p_id:.4g} | CD(Nemenyi)={cd:.3f}")
    print(f"Modelo de referencia (mayor media dice_test): {ref} "
          f"({medias[ref_i]:.4f})")

    # matrices de imagenes por modelo/semilla (para bootstrap jerarquico)
    imgs = {(m, s): cargar_imagenes(res_dir, m, s, "test")["dice"].values
            for m in elegidos for s in a.semillas}
    matriz_ref = [imgs[(ref, s)] for s in a.semillas]

    # --------------------------------------------------------- Tabla B
    filas_B, pvalores, parejas = [], [], []
    for i, m in enumerate(elegidos):
        lo, hi = ic_bootstrap_semillas(muestras[i], rng)
        if m == ref:
            filas_B.append({
                "modelo": m, "dice_test_media": round(medias[i], 5),
                "dice_test_desv": round(desv[i], 5),
                "ic95_lo": round(lo, 5), "ic95_hi": round(hi, 5),
                "rank_medio": round(rank_medio[i], 2),
                "dice_val_media": round(float(np.mean(
                    [datos[(m, s)]["dice_val"] for s in a.semillas])), 5),
                "ref": "referencia",
                "dif_vs_ref": 0.0, "ic95_dif_lo": None,
                "ic95_dif_hi": None, "p_wilcoxon": None, "p_holm": None,
                "veredicto": "referencia (mayor media)",
            })
            continue
        matriz_m = [imgs[(m, s)] for s in a.semillas]
        d_lo, d_hi = dif_jerarquica(matriz_ref, matriz_m, rng)
        difs = np.array(matriz_ref).mean(axis=1) - np.array(
            matriz_m).mean(axis=1)          # ref - modelo por semilla
        try:
            _, p_exacto = stats.wilcoxon(difs, method="exact")
        except ValueError:
            p_exacto = 1.0
        pvalores.append(p_exacto)
        parejas.append((m, difs, (d_lo, d_hi), p_exacto))
        filas_B.append({
            "modelo": m, "dice_test_media": round(medias[i], 5),
            "dice_test_desv": round(desv[i], 5),
            "ic95_lo": round(lo, 5), "ic95_hi": round(hi, 5),
            "rank_medio": round(rank_medio[i], 2),
            "dice_val_media": round(float(np.mean(
                [datos[(m, s)]["dice_val"] for s in a.semillas])), 5),
            "dif_vs_ref": round(float(difs.mean()), 5),
            "ic95_dif_lo": round(d_lo, 5), "ic95_dif_hi": round(d_hi, 5),
            "p_wilcoxon": round(p_exacto, 6),
            "veredicto": None,
        })

    if pvalores:
        ph = holm(pvalores)
        p_holm = {m: float(ph[i]) for i, (m, _d, _ic, _p) in
                  enumerate(parejas)}
        for fila in filas_B:
            if fila["p_wilcoxon"] is None:
                continue
            fila["p_holm"] = round(p_holm[fila["modelo"]], 6)
            lo_d, hi_d = fila["ic95_dif_lo"], fila["ic95_dif_hi"]
            if lo_d > a.delta:
                v = f"peor que {ref} por >= delta"
            elif hi_d < -a.delta:
                v = f"mejor que {ref} por >= delta"
            elif lo_d > -a.delta and hi_d < a.delta:
                v = f"equivalente a {ref} (|dif| < delta)"
            else:
                v = "indeterminado (IC cruza +/- delta)"
            fila["veredicto"] = v
    pd.DataFrame(filas_B).to_csv(os.path.join(out, "tabla_B_inferencia.csv"),
                                 index=False)

    # --------------------------------------------------------- Tabla C
    perfil = {}
    ruta_perfil = os.path.join(res_dir, "perfil_modelos.csv")
    if os.path.exists(ruta_perfil):
        with open(ruta_perfil, encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                perfil[fila["modelo"]] = fila
    filas_C = []
    for i, m in enumerate(elegidos):
        segs = [datos[(m, s)].get("segundos") for s in a.semillas]
        segs = [x for x in segs if x]
        p = perfil.get(m, {})
        filas_C.append({
            "modelo": m, "dice_test_media": round(medias[i], 5),
            "params_m": p.get("params_m"), "gmacs": p.get("gmacs"),
            "latencia_ms_cpu": p.get("latencia_ms_mediana"),
            "pico_memoria_mb": p.get("pico_memoria_mb") or None,
            "tiempo_train_medio_s": (round(float(np.mean(segs)), 1)
                                     if segs else None),
        })
    # frontera de Pareto: maximizar dice, minimizar GMACs (coste primario)
    dfC = pd.DataFrame(filas_C)
    en_frente = []
    for i, fila in dfC.iterrows():
        if fila["gmacs"] is None or pd.isna(fila["gmacs"]):
            en_frente.append(False)
            continue
        # j domina a i si j es mejor o igual en ambos ejes y estrictamente
        # mejor en al menos uno (dice mayor, GMACs menor)
        domina = ((dfC["dice_test_media"] >= fila["dice_test_media"]) &
                  (dfC["gmacs"] <= fila["gmacs"]) &
                  ((dfC["dice_test_media"] > fila["dice_test_media"]) |
                   (dfC["gmacs"] < fila["gmacs"]))).any()
        en_frente.append(not bool(domina))
    dfC["en_frente_pareto"] = en_frente
    dfC.to_csv(os.path.join(out, "tabla_C_coste.csv"), index=False)

    # ---------------------------------------------------- inferencia.json
    resumen = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "delta": a.delta, "alpha": a.alpha, "n_semillas": n,
        "n_modelos": k, "modelos": elegidos, "excluidos": excluidos,
        "duplicados_sospechosos": duplicados,
        "friedman": {"chi2": round(float(chi2), 4), "p": float(p_friedman),
                     "iman_davenport_F": round(float(f_id), 4),
                     "p_iman": float(p_id)},
        "nemenyi_cd": round(float(cd), 4),
        "q_alpha": round(float(q), 4),
        "referencia": ref,
        "rank_medio": {m: round(float(rank_medio[i]), 2)
                       for i, m in enumerate(elegidos)},
        "media_desv": {m: [round(float(medias[i]), 5),
                           round(float(desv[i]), 5)]
                       for i, m in enumerate(elegidos)},
        "wilcoxon_holm": [
            {"modelo": m, "dif_media_ref_menos_m": round(float(d.mean()), 5),
             "p_exacto": round(float(p), 6), "p_holm": round(float(ph[i]), 6),
             "ic95_dif": [round(x, 5) for x in ic]}
            for i, (m, d, ic, p) in enumerate(parejas)] if pvalores else [],
    }
    with open(os.path.join(out, "inferencia.json"), "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\nEscrito: {out}")
    for nombre in ("tabla_A_runs.csv", "tabla_B_inferencia.csv",
                   "tabla_C_coste.csv", "matriz_test.csv",
                   "inferencia.json"):
        print(f"  - {nombre}")


if __name__ == "__main__":
    main()
