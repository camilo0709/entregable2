# Comparaciones del modelo original (E3)

Original: `lacca_unet`; rivales: lacca_eje_fijo, lacca_sin_asimetria, lacca_global. Semillas por modelo: {'lacca_unet': 5, 'lacca_eje_fijo': 5, 'lacca_sin_asimetria': 5, 'lacca_global': 5}.

Friedman (imágenes): chi2=2.681, p=0.4435

| rival | prueba | Δ Dice | p | p Holm | Cliff δ | IC 95 % | gana |
|---|---|---|---|---|---|---|---|
| lacca_eje_fijo | T1 Wilcoxon (imagen) | -0.0029 | 0.6601 | 1 | -0.013 |  | no |
| lacca_eje_fijo | T2 Bootstrap pareado | -0.0029 | 0.6338 | 1 | -0.013 | [-0.0148, 0.0086] | no |
| lacca_eje_fijo | T3 Mann-Whitney (semillas) | -0.0029 | 1 | 1 | +0.040 |  | no |
| lacca_sin_asimetria | T1 Wilcoxon (imagen) | +0.0008 | 0.9338 | 1 | -0.003 |  | no |
| lacca_sin_asimetria | T2 Bootstrap pareado | +0.0008 | 0.8888 | 1 | -0.003 | [-0.0117, 0.0135] | no |
| lacca_sin_asimetria | T3 Mann-Whitney (semillas) | +0.0008 | 1 | 1 | +0.040 |  | no |
| lacca_global | T1 Wilcoxon (imagen) | +0.0109 | 0.3958 | 1 | +0.024 |  | no |
| lacca_global | T2 Bootstrap pareado | +0.0109 | 0.2508 | 1 | +0.024 | [-0.0080, 0.0294] | no |
| lacca_global | T3 Mann-Whitney (semillas) | +0.0109 | 0.5476 | 1 | +0.280 |  | no |

Holm sobre los 9 p-valores. 'gana' = significativo tras Holm, |δ| >= 0.147 y diferencia a favor del original.