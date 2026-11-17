"""Construye las tablas livianas del tablero a partir de resultados/e3/ (sin reentrenar)."""
import json, glob, os
import numpy as np, pandas as pd
R = "resultados/e3"; O = "data"; os.makedirs(O, exist_ok=True)
FAM = {"unet": "CNN", "attention_unet": "CNN con atención", "unet_pp": "CNN anidada",
       "deeplabv3p": "CNN atrous", "transunet": "Híbrido CNN-Transformer", "swin_unet": "Transformer",
       "unetr": "Transformer", "segformer": "Transformer", "mednext": "CNN moderna",
       "lacca_unet": "Original (LACCA)", "lacca_eje_fijo": "Original (ablación)",
       "lacca_sin_asimetria": "Original (ablación)", "lacca_global": "Original (ablación)",
       "unet_fuerte": "Baseline (U-Net reforzada)"}
t = pd.read_csv(f"{R}/resumen_test.csv")
g = t.groupby("modelo").agg(n_semillas=("semilla", "nunique"), dice_media=("dice_test", "mean"),
    dice_sd=("dice_test", "std"), iou=("iou_test", "mean"), hd95=("hd95_test", "mean"),
    assd=("assd_test", "mean")).reset_index()
p = pd.read_json(f"{R}/onnx/perfil_modelos.json").T.reset_index(drop=True)
p = p[["modelo", "parametros", "gflops_1img", "latencia_ms_ort_cpu", "sha256_onnx"]]
g = g.merge(p, on="modelo", how="left")
cal = []
for d in sorted(glob.glob(f"{R}/extra/*_s42")):
    m = os.path.basename(d)[:-4]; c = json.load(open(f"{d}/calibracion.json"))
    cal.append({"modelo": m, "T": c["T"], "ece_antes": c["ece_antes"], "ece_despues": c["ece_despues"],
                "brier_antes": c["brier_antes"], "brier_despues": c["brier_despues"]})
    r = pd.read_csv(f"{d}/robustez.csv"); r["modelo"] = m
    r.to_csv(f"{O}/_rob_{m}.csv", index=False)
    pd.read_csv(f"{d}/confiabilidad.csv").assign(modelo=m).to_csv(f"{O}/_conf_{m}.csv", index=False)
cal = pd.DataFrame(cal); cal.to_csv(f"{O}/calibracion.csv", index=False)
g = g.merge(cal[["modelo", "ece_despues"]].rename(columns={"ece_despues": "ece"}), on="modelo", how="left")
g["familia"] = g.modelo.map(FAM)
g = g.sort_values("dice_media", ascending=False)
g.round(5).to_csv(f"{O}/leaderboard.csv", index=False)
pd.concat([pd.read_csv(f) for f in glob.glob(f"{O}/_rob_*.csv")]).to_csv(f"{O}/robustez.csv", index=False)
pd.concat([pd.read_csv(f) for f in glob.glob(f"{O}/_conf_*.csv")]).to_csv(f"{O}/confiabilidad.csv", index=False)
for f in glob.glob(f"{O}/_*.csv"): os.remove(f)
pd.read_csv(f"{R}/analisis_e3/comparaciones_e3.csv").to_csv(f"{O}/comparaciones_original.csv", index=False)
pd.read_csv("experiments/dice_por_imagen_media_semillas.csv").to_csv(f"{O}/dice_por_imagen.csv", index=False)
# EDA
tr = pd.read_csv("datos/splits/train.csv"); va = pd.read_csv("datos/splits/val.csv")
e = pd.concat([tr, va])[["id", "split", "tb_type", "n_boxes", "mask_frac"]]
e.to_csv(f"{O}/eda_train_val.csv", index=False)
sp = pd.DataFrame({"split": ["train", "val", "test_pos", "test_neg"],
    "n": [len(tr), len(va), len(pd.read_csv("datos/splits/test_pos.csv")), len(pd.read_csv("datos/splits/test_neg.csv"))]})
sp.to_csv(f"{O}/eda_splits.csv", index=False)
print(g[["modelo", "n_semillas", "dice_media", "dice_sd", "parametros", "latencia_ms_ort_cpu"]].round(4).to_string())
print(sp)
