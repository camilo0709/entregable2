"""
sobreajuste.py — Diagnóstico de sobreajuste y optimismo de selección.

TRES PREGUNTAS
    1. ¿La pérdida de entrenamiento es mucho menor que la de validación en el
       checkpoint elegido? (brecha train–val)
    2. ¿Cuánto está inflado el Dice de validación por haber elegido la mejor
       época entre decenas? (optimismo de selección)
    3. ¿La validación se sostiene en test? (brecha val–test, cuando `dice_test`
       ya existe)

QUÉ NO HACE
    No reentrena ni recalibra. La brecha se mide con el checkpoint congelado y
    la pérdida oficial (DiceBCE 0,5/0,5). El optimismo de selección se estima
    con la curva de validación registrada, no con un .632+ que exigiría
    reajustar el modelo en cada remuestreo (se declara como limitación).

NOTA SOBRE `dice_val`
    El `dice_val` de `resultado.json` se calculó durante el entrenamiento
    (Dice suave por lote); el que reporta este módulo es Dice duro por imagen
    a umbral 0,5. Pueden diferir. La brecha train–val usa la **misma**
    definición (Dice duro) en ambos splits, que es lo que importa para el
    diagnóstico.

CRITERIO (post-hoc, declarado)
    margen de brecha de pérdida = 0,05 (el mismo de la Tabla B de la Sección 7)
        gap <= 0,05            -> sin señal de sobreajuste
        0,05 < gap <= 0,10     -> sobreajuste moderado
        gap > 0,10             -> sobreajuste alto
    La brecha de Dice se reporta como diagnóstico secundario (puntos de Dice).

USO
    python codigo/sobreajuste.py --modelo deeplabv3p --semilla 42 --limite 8
    python codigo/sobreajuste.py --modelo deeplabv3p --semilla 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_tb import DatasetTB                    # noqa: E402
from driver_entrenamiento import DiceBCELoss        # noqa: E402
from evaluar_test import cargar_modelo              # noqa: E402
from metricas_tb import dice                        # noqa: E402

MARGEN = 0.05


def _logits(net, x):
    salida = net(x)
    if isinstance(salida, (tuple, list)):
        for s in salida:
            if isinstance(s, torch.Tensor) and s.ndim == 4:
                return s
        raise TypeError("salida del modelo sin tensor 4D")
    return salida


@torch.no_grad()
def evaluar_split(net, split, device, criterio, limite=None, batch=8):
    """Pérdida media y Dice medio por imagen en un split."""
    ds = DatasetTB(split, esperar_gt=True, verbose=False)
    if limite is not None:
        ds.filas = ds.filas[:limite]
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    perdidas, pesos, dices = [], [], []
    for imgs, msk, _ in dl:
        imgs, msk = imgs.to(device), msk.to(device)
        logits = _logits(net, imgs)
        perdidas.append(float(criterio(logits, msk).item()))
        pesos.append(imgs.shape[0])
        prob = torch.sigmoid(logits)[:, 0].cpu().numpy()
        gt = msk.cpu().numpy()[:, 0] > 0.5
        for i in range(prob.shape[0]):
            dices.append(float(dice(prob[i] > 0.5, gt[i])))
    return {
        "split": split,
        "n": len(dices),
        "loss": float(np.average(perdidas, weights=pesos)),
        "dice": float(np.mean(dices)),
        "dices": dices,
    }


def optimismo_seleccion(historial):
    """Inflación del mejor Dice de validación respecto de la curva completa.

    Devuelve también el máximo esperado de n normales estándar (aprox. de
    Cramér–von Mises) para juzgar si el pico es compatible con azar.
    """
    vals = [h["dice_val"] for h in historial
            if h.get("dice_val") is not None]
    if len(vals) < 3:
        return None
    vals = np.asarray(vals, dtype=float)
    mejor, media = float(vals.max()), float(vals.mean())
    sd = float(vals.std(ddof=1))
    n = len(vals)
    if sd > 0:
        z = (mejor - media) / sd
        emax = (math.sqrt(2 * math.log(n))
                - (math.log(math.log(n)) + math.log(4 * math.pi))
                / (2 * math.sqrt(2 * math.log(n))))
    else:
        z, emax = None, None
    return {
        "n_epocas": n,
        "mejor_dice_val": round(mejor, 6),
        "media_dice_val": round(media, 6),
        "sd_dice_val": round(sd, 6),
        "optimismo": round(mejor - media, 6),
        "z_pico": None if z is None else round(z, 4),
        "z_max_esperado": None if emax is None else round(emax, 4),
    }


def diagnosticar(clave, semilla, res_dir, device="cpu", limite=None):
    rd = os.path.join(res_dir, clave, f"seed_{semilla}")
    ckpt = os.path.join(rd, "mejor_modelo.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"sin checkpoint: {ckpt}")
    with open(os.path.join(rd, "resultado.json"), encoding="utf-8") as f:
        r = json.load(f)

    net = cargar_modelo(clave, ckpt, torch.device(device))
    crit = DiceBCELoss(0.5, 0.5)
    tr = evaluar_split(net, "train", torch.device(device), crit, limite)
    va = evaluar_split(net, "val", torch.device(device), crit, limite)

    gap_loss = va["loss"] - tr["loss"]
    gap_dice = tr["dice"] - va["dice"]
    if gap_loss <= MARGEN:
        estado = "sin señal"
    elif gap_loss <= 2 * MARGEN:
        estado = "moderado"
    else:
        estado = "alto"

    out = {
        "modelo": clave,
        "semilla": semilla,
        "limite_por_split": limite,
        "train_loss": round(tr["loss"], 6),
        "val_loss": round(va["loss"], 6),
        "gap_loss": round(gap_loss, 6),
        "margen": MARGEN,
        "estado": estado,
        "train_dice": round(tr["dice"], 6),
        "val_dice": round(va["dice"], 6),
        "gap_dice": round(gap_dice, 6),
        "optimismo_seleccion": optimismo_seleccion(r.get("historial", [])),
        "dice_val_json": r.get("dice_val"),
        "dice_test_json": r.get("dice_test"),
    }
    if r.get("dice_test") is not None and r.get("dice_val") is not None:
        out["gap_val_test"] = round(r["dice_val"] - r["dice_test"], 6)
    return out


def main():
    ap = argparse.ArgumentParser(description="Brecha train–val y optimismo")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--semilla", type=int, required=True)
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limite", type=int, default=None,
                    help="N imágenes por split (prueba rápida)")
    ap.add_argument("--guardar", action="store_true")
    args = ap.parse_args()

    r = diagnosticar(args.modelo, args.semilla, args.resultados,
                     args.device, args.limite)
    print(f"\n{args.modelo} · semilla {args.semilla} · "
          f"({r['limite_por_split'] or 'todos'} por split)")
    print(f"  pérdida: train {r['train_loss']:.5f} · val {r['val_loss']:.5f} "
          f"· brecha {r['gap_loss']:+.5f} (margen {r['margen']}) -> "
          f"{r['estado']}")
    print(f"  Dice:    train {r['train_dice']:.5f} · val {r['val_dice']:.5f} "
          f"· brecha {r['gap_dice']:+.5f}")
    o = r["optimismo_seleccion"]
    if o:
        print(f"  selección: mejor val {o['mejor_dice_val']:.5f} vs media "
              f"{o['media_dice_val']:.5f} -> optimismo {o['optimismo']:+.5f} "
              f"(z={o['z_pico']}, máx. esperado≈{o['z_max_esperado']} "
              f"en {o['n_epocas']} épocas)")
    if "gap_val_test" in r:
        print(f"  val-test: {r['gap_val_test']:+.5f}")
    else:
        print("  val-test: pendiente (dice_test aún no calculado)")

    if args.guardar:
        out = Path(args.resultados) / "sobreajuste"
        out.mkdir(parents=True, exist_ok=True)
        ruta = out / f"{args.modelo}_s{args.semilla}.json"
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"  guardado: {ruta}")


if __name__ == "__main__":
    main()
