# §4 · Implementación de modelos benchmark

> **Estado:** entrenamiento en curso en Colab (25 % de la rúbrica).

Contendrá:

- **Diseño del benchmark:** 10 modelos × 5 semillas `[42, 123, 777, 2024, 31415]`,
  presupuesto idéntico (épocas/datos/hardware/métrica), mismos splits sellados.
- **Arquitecturas** (Top 10 de la [§3](03_revision_sistematica.md)): U-Net,
  Attention U-Net, U-Net++, DeepLabV3+, TransUNet, Swin-Unet, UNETR, SegFormer,
  MedNeXt y nnU-Net (pipeline propio).
- **Baselines:** `bbox_mask` (no profundo) y `empty_mask` (trivial).
- **Protocolo de ejecución:** driver reanudable (`codigo/driver_entrenamiento.py`),
  pipeline nnU-Net (`codigo/entrenamiento_nnunet.py`, equivalencia folds↔semillas
  declarada en `docs/NNUNET_PIPELINE.md`).
- **Resultados por corrida:** `resultados/<modelo>/seed_<s>/resultado.json` +
  `por_imagen.csv` (val) + `por_imagen_test.csv` (test_pos GT plata).

Fuentes: `codigo/modelos_tb.py`, `codigo/setup_colab.ipynb`, `docs/NNUNET_PIPELINE.md`.
