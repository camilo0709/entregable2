"""
Evalúa checkpoints entrenados en un dataset externo sobre su test retenido.

Entrenamiento (driver):  --splits-dir datos/externos_splits/<nombre>
Evaluación (este script): lee <splits>/heldout/val.csv  (15 % nunca visto)
Escribe por run: <res>/<modelo>/seed_<s>/por_imagen_externo.csv y
resumen: <res>/resumen_externo.csv (modelo, semilla, dice, iou medios).
Uso: python evaluar_externos.py --resultados R/externos/montgomery \\
        --splits-dir datos/externos_splits/montgomery --modelos unet lacca_unet
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelos_tb import SEMILLAS, SEMILLAS_EXTRA, crear_modelo  # noqa: E402
import metricas_tb as M                                           # noqa: E402


def evaluar(res_dir, splits_dir, modelo, semilla, device):
    from dataset_tb import DatasetTB
    ck_p = os.path.join(res_dir, modelo, f"seed_{semilla}", "mejor_modelo.pt")
    if not os.path.exists(ck_p):
        return None
    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    net, _ = crear_modelo(modelo, in_channels=1, out_channels=1, verbose=False,
                          params_extra=(ck.get("config") or {}).get("params_modelo"))
    net.load_state_dict(ck.get("state_dict", ck))
    net = net.to(device).eval()
    ds = DatasetTB("val", splits_dir=os.path.join(splits_dir, "heldout"),
                   augmentar=False, verbose=False)
    filas = []
    with torch.no_grad():
        for xb, yb, info in DataLoader(ds, batch_size=4, num_workers=0):
            pr = (torch.sigmoid(net(xb.to(device))) > 0.5).cpu().numpy().astype(np.uint8)
            for k in range(xb.shape[0]):
                m = M.metricas_imagen(pr[k, 0], (yb[k, 0].numpy() > 0.5).astype(np.uint8))
                m["id"] = info["id"][k]
                filas.append(m)
    ruta = os.path.join(res_dir, modelo, f"seed_{semilla}", "por_imagen_externo.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return {"modelo": modelo, "semilla": semilla, "n": len(filas),
            "dice": float(np.mean([x["dice"] for x in filas])),
            "iou": float(np.mean([x["iou"] for x in filas]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--modelos", nargs="+", required=True)
    ap.add_argument("--semillas", nargs="*", type=int, default=SEMILLAS + SEMILLAS_EXTRA)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = []
    for m in a.modelos:
        for s in a.semillas:
            r = evaluar(a.resultados, a.splits_dir, m, s, dev)
            if r:
                res.append(r)
                print(f"{m:16s} s{s:<6d} dice={r['dice']:.4f} iou={r['iou']:.4f}")
    if res:
        with open(os.path.join(a.resultados, "resumen_externo.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)


if __name__ == "__main__":
    main()
