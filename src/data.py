"""Carga y caché de las tablas livianas del tablero (data/*.csv)."""
from functools import lru_cache
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NOMBRES = {"unet": "U-Net", "attention_unet": "Attention U-Net", "unet_pp": "U-Net++",
           "deeplabv3p": "DeepLabV3+", "transunet": "TransUNet", "swin_unet": "Swin-UNet",
           "unetr": "UNETR", "segformer": "SegFormer", "mednext": "MedNeXt",
           "lacca_unet": "LACCA-UNet (original)", "lacca_eje_fijo": "LACCA · eje fijo",
           "lacca_sin_asimetria": "LACCA · sin asimetría", "lacca_global": "LACCA · atención global",
           "unet_fuerte": "U-Net reforzada"}


@lru_cache(maxsize=None)
def tabla(nombre: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{nombre}.csv")


def leaderboard() -> pd.DataFrame:
    d = tabla("leaderboard").copy()
    d["Modelo"] = d["modelo"].map(NOMBRES)
    return d


@lru_cache(maxsize=None)
def perfil() -> dict:
    return json.loads((ROOT / "artifacts" / "perfil_modelos.json").read_text(encoding="utf-8"))


def modelos_con_onnx() -> list:
    return list(perfil().keys())
