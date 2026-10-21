"""
================================================================================
PERFIL COMPUTACIONAL DE LOS MODELOS (rúbrica 4.2: params, FLOPs, latencia)
================================================================================

PROPOSITO:
    Medir, con el MISMO protocolo para todas las arquitecturas:
      - numero de parametros (total y entrenable)
      - MACs y FLOPs a 512x512x1 (batch 1) via thop  (FLOPs = 2 x MACs)
      - latencia de inferencia (mediana de N pases, 1 imagen)
      - pico de memoria GPU (solo con --device cuda)

SALIDA:
    resultados/perfil_modelos.csv  + tabla markdown por consola.

USO:
    Local (CPU):   python codigo/perfil_modelos.py
    Colab (GPU):   python codigo/perfil_modelos.py --device cuda

NOTAS:
    - nnU-Net no se perfila aqui: es un pipeline con su propia configuracion
      (folds, patch size autoconfigurado); se reporta por separado.
    - empty_mask / bbox_mask no tienen red: params/FLOPs = 0.
    - Los numeros de params/FLOPs son exactos (no dependen del hardware);
      la latencia y la memoria si dependen del dispositivo declarado.
================================================================================
"""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelos_tb import crear_modelo, listar_modelos, MODELOS   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = (1, 1, 512, 512)


def perfilar(clave, device, n_pases=5):
    """Devuelve el dict de perfil de UNA red (None si no aplica)."""
    meta = MODELOS[clave]
    if meta["backend"] == "baseline":
        return {"modelo": clave, "notas": "baseline sin red (params=0)"}
    if meta["backend"] == "nnunet":
        return {"modelo": clave,
                "notas": "pipeline propio (ver docs/NNUNET_PIPELINE.md)"}

    net, _ = crear_modelo(clave, in_channels=1, out_channels=1, verbose=False)
    net = net.to(device).eval()

    params_total = sum(p.numel() for p in net.parameters())
    params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)

    x = torch.randn(*ENTRADA, device=device)

    # ------------------------------------------------- memoria (antes de thop)
    # Se mide primero porque thop deja hooks enganchados si falla y ensucia
    # los forwards posteriores.
    pico_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            net(x)
        torch.cuda.synchronize()
        pico_mb = torch.cuda.max_memory_allocated() / 1e6

    # ------------------------------------------------------------- latencia
    with torch.no_grad():
        if device.type == "cuda":
            net(x)                                  # warm-up + compilación
            torch.cuda.synchronize()
        else:
            net(x)                                  # warm-up
        tiempos = []
        for _ in range(n_pases):
            t0 = time.perf_counter()
            net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tiempos.append(time.perf_counter() - t0)
    lat_ms = sorted(tiempos)[len(tiempos) // 2] * 1000.0   # mediana

    # ----------------------------------------------------------------- FLOPs
    # thop al final: si falla (hooks incompatibles con algún módulo), el
    # modelo ya no se vuelve a usar.
    macs = None
    with torch.no_grad():
        try:
            from thop import profile as thop_profile
            macs, _ = thop_profile(net, inputs=(x,), verbose=False)
        except Exception as e:
            print(f"  (thop no pudo medir {clave}: {type(e).__name__})")

    return {
        "modelo": clave,
        "params_m": round(params_total / 1e6, 2),
        "params_entrenables_m": round(params_train / 1e6, 2),
        "gmacs": (round(macs / 1e9, 2) if macs else None),
        "gflops": (round(2 * macs / 1e9, 2) if macs else None),
        "latencia_ms_mediana": round(lat_ms, 1),
        "pico_memoria_mb": (round(pico_mb, 1) if pico_mb else None),
        "device": str(device),
        "notas": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--salida", default=os.path.join(ROOT, "resultados",
                                                     "perfil_modelos.csv"))
    a = ap.parse_args()

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda pedido pero no disponible")

    claves = [k for k in listar_modelos() if k != "nnunet"] + ["nnunet"]
    filas = []
    print(f"Perfil computacional @ {ENTRADA[1]}x{ENTRADA[2]}x{ENTRADA[3]}, "
          f"device={device}")
    for k in claves:
        print(f"  perfilando {k} ...", flush=True)
        filas.append(perfilar(k, device))

    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    campos = ["modelo", "params_m", "params_entrenables_m", "gmacs", "gflops",
              "latencia_ms_mediana", "pico_memoria_mb", "device", "notas"]
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for fila in filas:
            w.writerow(fila)

    # --------------------------------------------------- tabla markdown
    print("\n| modelo | params (M) | GMACs | GFLOPs | latencia (ms) | "
          "pico mem (MB) | notas |")
    print("|---|---|---|---|---|---|---|")
    for r in filas:
        print(f"| {r['modelo']} | {r.get('params_m', '—')} | "
              f"{r.get('gmacs', '—')} | {r.get('gflops', '—')} | "
              f"{r.get('latencia_ms_mediana', '—')} | "
              f"{r.get('pico_memoria_mb', '—')} | {r.get('notas', '')} |")
    print(f"\nCSV: {a.salida}")


if __name__ == "__main__":
    main()
