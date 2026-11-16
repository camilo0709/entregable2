"""
EXPORTACIÓN A ONNX + PERFIL (Entregable 3, Bloque B)

Por modelo (semilla de referencia, por defecto 42):
  - carga mejor_modelo.pt (con los hiperparámetros del checkpoint),
  - exporta resultados/onnx/<modelo>.onnx (entrada fija 1x1x512x512),
  - ida y vuelta: |sigmoid(torch) - sigmoid(onnxruntime)| < tol (1e-4) sobre
    entradas aleatorias y, si se da --splits-dir, 4 imágenes reales de val,
  - metadata.json: nombre, semilla, dice_val, forma, normalización, umbral,
    sha256, versiones; y perfil: parámetros, FLOPs (1 imagen), latencia
    mediana ORT-CPU y torch-GPU (si hay), memoria pico GPU.

Uso:  python exportar_onnx.py --resultados resultados_e3 --modelos unet lacca_unet
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelos_tb import MODELOS, crear_modelo, listar_modelos  # noqa: E402

TAM = 512


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cargar_red(res_dir, modelo, semilla):
    ck_path = os.path.join(res_dir, modelo, f"seed_{semilla}", "mejor_modelo.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net, _ = crear_modelo(modelo, in_channels=1, out_channels=1, verbose=False,
                          params_extra=(ck.get("config") or {}).get("params_modelo"))
    net.load_state_dict(ck.get("state_dict", ck))
    return net.eval(), ck, ck_path


def exportar(net, ruta, tam=TAM):
    x = torch.randn(1, 1, tam, tam)
    kw = dict(opset_version=17, input_names=["x"], output_names=["logits"])
    try:
        torch.onnx.export(net, (x,), ruta, dynamo=False, **kw)
    except TypeError:                       # torch < 2.5 no tiene 'dynamo'
        torch.onnx.export(net, (x,), ruta, **kw)


def ida_y_vuelta(net, ruta, entradas, tol):
    import onnxruntime as ort
    s = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    peor = 0.0
    for x in entradas:
        with torch.no_grad():
            a = torch.sigmoid(net(x)).numpy()
        b = 1 / (1 + np.exp(-s.run(None, {"x": x.numpy()})[0]))
        peor = max(peor, float(np.abs(a - b).max()))
    return peor, peor < tol


def perfil(net, ruta, tam=TAM, reps=15):
    x = torch.randn(1, 1, tam, tam)
    out = {"parametros": sum(p.numel() for p in net.parameters())}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            net(x)
        out["gflops_1img"] = round(fc.get_total_flops() / 1e9, 2)
    except Exception as e:                                 # noqa: BLE001
        out["gflops_1img"] = None
        out["gflops_error"] = str(e)[:80]
    import onnxruntime as ort
    s = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    xn = x.numpy()
    s.run(None, {"x": xn})
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        s.run(None, {"x": xn})
        t.append((time.perf_counter() - t0) * 1000)
    out["latencia_ms_ort_cpu"] = round(float(np.median(t)), 1)
    if torch.cuda.is_available():
        g = net.to("cuda")
        xg = x.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            g(xg)
            torch.cuda.synchronize()
            t = []
            for _ in range(reps):
                t0 = time.perf_counter()
                g(xg)
                torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) * 1000)
        out["latencia_ms_torch_gpu"] = round(float(np.median(t)), 1)
        out["memoria_pico_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 0)
        net.to("cpu")
    return out


def imagenes_val(splits_dir, n=4):
    from dataset_tb import DatasetTB
    ds = DatasetTB("val", splits_dir=splits_dir, augmentar=False, verbose=False)
    return [ds[i][0].unsqueeze(0) for i in range(min(n, len(ds)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--modelos", nargs="*", default=None)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    modelos = a.modelos or [m for m in listar_modelos()
                            if MODELOS[m].get("entrenable") and MODELOS[m]["backend"] != "nnunet"]
    sal = a.salida or os.path.join(a.resultados, "onnx")
    os.makedirs(sal, exist_ok=True)
    reales = imagenes_val(a.splits_dir) if a.splits_dir else []
    resumen = {}
    for m in modelos:
        print(f"\n[{m}]", flush=True)
        try:
            net, ck, ckp = cargar_red(a.resultados, m, a.semilla)
            ruta = os.path.join(sal, f"{m}.onnx")
            exportar(net, ruta)
            ent = [torch.randn(1, 1, TAM, TAM) for _ in range(2)] + reales
            dif, ok = ida_y_vuelta(net, ruta, ent, a.tol)
            meta = {"modelo": m, "semilla": a.semilla, "dice_val": ck.get("dice_val"),
                    "entrada": [1, 1, TAM, TAM], "salida": "logits (aplicar sigmoide)",
                    "normalizacion": "z-score por imagen", "umbral": 0.5,
                    "opset": 17, "torch": torch.__version__,
                    "sha256_checkpoint": sha256(ckp), "sha256_onnx": sha256(ruta),
                    "tolerancia_probabilidad": a.tol,
                    "dif_max_probabilidad": dif, "ida_y_vuelta_ok": ok,
                    "n_entradas_verificadas": len(ent), **perfil(net, ruta)}
            json.dump(meta, open(os.path.join(sal, f"{m}.metadata.json"), "w",
                                 encoding="utf-8"), indent=2, ensure_ascii=False)
            resumen[m] = meta
            print(f"  dif máx prob = {dif:.2e} {'OK' if ok else 'FALLA'} | "
                  f"{meta['parametros']/1e6:.1f} M par | {meta['gflops_1img']} GFLOPs | "
                  f"{meta['latencia_ms_ort_cpu']} ms CPU")
        except Exception as e:                              # noqa: BLE001
            resumen[m] = {"modelo": m, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print("  ERROR:", resumen[m]["error"])
    json.dump(resumen, open(os.path.join(sal, "perfil_modelos.json"), "w",
                            encoding="utf-8"), indent=2, ensure_ascii=False)
    malos = [m for m, v in resumen.items() if not v.get("ida_y_vuelta_ok")]
    print("\nFallaron o no cumplen tolerancia:", malos or "ninguno")


if __name__ == "__main__":
    main()
