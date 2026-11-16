"""
BÚSQUEDA DE HIPERPARÁMETROS - Entregable 3, Bloque B

Mismo presupuesto para todos los modelos: N_INTENTOS corridas por modelo
(el intento 0 es la configuración por defecto del protocolo, así la búsqueda
nunca deja un modelo peor que su baseline; los demás son búsqueda aleatoria
con semilla fija por modelo). Selección SOLO por dice_val (validación); el
test sellado no se toca.

Reanudable: cada intento escribe su resultado.json; al reejecutar se saltan
los terminados. Salidas en <resultados>/hpo/:
    hpo_intentos.csv      un renglón por intento (config + dice_val)
    mejor_config.json     {modelo: {lr, weight_decay, batch_size, params_modelo}}
                          -> se pasa al driver con --config-hpo

Uso (Colab, ver colab_01_hpo.ipynb):
    python hpo_busqueda.py --splits-dir ... --resultados ... --max-minutos 330
    python hpo_busqueda.py --modelos unet_fuerte lacca_unet --intentos 6 --dry-run
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_INTENTOS = 6
SEMILLA_HPO = [42]
# "unet" se entrena SIN búsqueda (config del protocolo) y "unet_fuerte" es la
# misma red CON búsqueda (enmienda E4): así dejan de ser duplicados.
# nnunet tiene su propio pipeline.
MODELOS_HPO = ["unet_fuerte", "attention_unet", "unet_pp", "deeplabv3p", "transunet",
               "swin_unet", "unetr", "segformer", "mednext", "lacca_unet"]


def _logunif(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def muestrear(modelo, k, defecto):
    """Config del intento k (k=0 -> defecto). Determinista por (modelo, k)."""
    if k == 0:
        cfg = dict(defecto)
        cfg["params_modelo"] = {}
        return cfg
    rng = random.Random(zlib.crc32(f"{modelo}:{k}".encode()))
    cfg = dict(defecto)            # hereda epochs, num_workers, paciencia, etc.
    cfg.update({"lr": round(_logunif(rng, 3e-5, 1e-3), 7),
                "weight_decay": round(_logunif(rng, 1e-6, 1e-3), 8),
                "batch_size": rng.choice([4, 8, 16]),
                "params_modelo": {}})
    if modelo == "lacca_unet":     # hiperparámetros propios del módulo
        cfg["params_modelo"] = {
            "radius": rng.choice([1, 2, 3]),
            "key_dim": rng.choice([8, 16, 32]),
            "value_dim": rng.choice([16, 32, 64]),
            "tau": rng.choice([0.05, 0.1, 0.3]),
            "eps": rng.choice([1e-4, 1e-3, 1e-2]),
            "theta_max": math.radians(rng.choice([10.0, 15.0, 20.0])),
            "gamma_init": rng.choice([0.0, 0.1]),
            "axis_init_std": rng.choice([0.0, 0.05]),
        }
    return cfg


def plan(modelos, n):
    return [(m, k) for m in modelos for k in range(n)]


def ruta_intento(hpo_dir, modelo, k):
    return os.path.join(hpo_dir, modelo, f"t{k:02d}")


def main():
    import driver_entrenamiento as D
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--modelos", nargs="*", default=MODELOS_HPO)
    ap.add_argument("--intentos", type=int, default=N_INTENTOS)
    ap.add_argument("--semillas-hpo", nargs="*", type=int, default=SEMILLA_HPO,
                    help="semillas por intento (por defecto UNA: 42, por costo)")
    ap.add_argument("--resultados", default=os.path.join(D.ROOT, "resultados"))
    ap.add_argument("--splits-dir", default=D.SPLITS_DIR)
    ap.add_argument("--max-minutos", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=D.CONFIG["epochs"])
    ap.add_argument("--num-workers", type=int, default=D.CONFIG["num_workers"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import torch
    hpo_dir = os.path.join(a.resultados, "hpo")
    os.makedirs(hpo_dir, exist_ok=True)
    base = dict(D.CONFIG, epochs=a.epochs, num_workers=a.num_workers)
    tareas = plan(a.modelos, a.intentos)

    def hecho(m, k):
        return all(D.run_completado(ruta_intento(hpo_dir, m, k), m, s)
                   for s in a.semillas_hpo)

    pend = [(m, k) for m, k in tareas if not hecho(m, k)]
    print(f"HPO: {len(tareas)} intentos ({len(a.modelos)} modelos x "
          f"{a.intentos}), pendientes {len(pend)}, semillas {a.semillas_hpo}")
    if a.dry_run:
        for m, k in pend:
            c = muestrear(m, k, base)
            print(f"  {m:15s} t{k:02d} lr={c['lr']:.2e} wd={c['weight_decay']:.1e} "
                  f"bs={c['batch_size']} {c['params_modelo']}")
        return
    if not torch.cuda.is_available() and not a.cpu:
        print("Sin GPU: abortado (usa Colab o --cpu para un smoke test).")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limite = a.max_minutos * 60 if a.max_minutos else None
    t0 = time.time()

    for m, k in pend:
        if limite and time.time() - t0 > limite:
            print("Límite de sesión: estado guardado, reejecuta el mismo comando.")
            break
        cfg = muestrear(m, k, base)
        print(f"\n[{m} t{k:02d}] lr={cfg['lr']:.2e} bs={cfg['batch_size']} "
              f"{cfg['params_modelo']}", flush=True)
        for s in a.semillas_hpo:
            rd = ruta_intento(hpo_dir, m, k)
            if D.run_completado(rd, m, s):
                continue
            r = D.entrenar_run(m, s, rd, cfg, device,
                               limite_segundos=(limite - (time.time() - t0))
                               if limite else None,
                               splits_dir=a.splits_dir)
            r["config"] = cfg
            if r.get("estado") == "completado":
                D.guardar_resultado(rd, m, s, r)
                # los pesos de un intento de HPO no se usan: ~100 MB c/u en Drive
                pesos = os.path.join(D.ruta_run(rd, m, s), "mejor_modelo.pt")
                if os.path.exists(pesos):
                    os.remove(pesos)
            else:
                print(f"   estado {r.get('estado')}: no se guarda como hecho")

    # ------------------------------------------------ consolidar resultados
    filas, mejor = [], {}
    for m in a.modelos:
        for k in range(a.intentos):
            ds = []
            for s in a.semillas_hpo:
                p = os.path.join(D.ruta_run(ruta_intento(hpo_dir, m, k), m, s),
                                 "resultado.json")
                if os.path.exists(p):
                    r = json.load(open(p, encoding="utf-8"))
                    if r.get("estado") == "completado":
                        ds.append(r["dice_val"])
            if len(ds) == len(a.semillas_hpo):
                cfg = muestrear(m, k, base)
                dv = sum(ds) / len(ds)
                filas.append({"modelo": m, "intento": k, "dice_val": round(dv, 5),
                              "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
                              "batch_size": cfg["batch_size"],
                              "params_modelo": json.dumps(cfg["params_modelo"])})
                if m not in mejor or dv > mejor[m][0]:
                    mejor[m] = (dv, cfg, k)
    if filas:
        with open(os.path.join(hpo_dir, "hpo_intentos.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
    salida = {m: {"lr": c["lr"], "weight_decay": c["weight_decay"],
                  "batch_size": c["batch_size"],
                  "params_modelo": c["params_modelo"],
                  "_intento": k, "_dice_val_hpo": round(dv, 5)}
              for m, (dv, c, k) in mejor.items()}
    with open(os.path.join(hpo_dir, "mejor_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nmejor_config.json con {len(salida)} modelos en {hpo_dir}")
    for m, v in salida.items():
        print(f"  {m:15s} intento {v['_intento']} dice_val={v['_dice_val_hpo']:.4f}")


if __name__ == "__main__":
    main()
