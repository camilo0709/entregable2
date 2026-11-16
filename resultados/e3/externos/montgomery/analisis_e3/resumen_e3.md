# Comparaciones del modelo original (E3)

Original: `lacca_unet`; rivales: attention_unet, unet_pp, mednext. Semillas por modelo: {'lacca_unet': 5, 'attention_unet': 5, 'unet_pp': 5, 'mednext': 5}.

Friedman (imágenes): chi2=31.473, p=6.759e-07

| rival | prueba | Δ Dice | p | p Holm | Cliff δ | IC 95 % | gana |
|---|---|---|---|---|---|---|---|
| attention_unet | T1 Wilcoxon (imagen) | +0.0028 | 9.06e-06 | 8.154e-05 | +0.186 |  | sí |
| attention_unet | T2 Bootstrap pareado | +0.0028 | 0.0001 | 0.0008 | +0.186 | [0.0017, 0.0042] | sí |
| attention_unet | T3 Mann-Whitney (semillas) | +0.0028 | 0.05556 | 0.2778 | +0.760 |  | no |
| unet_pp | T1 Wilcoxon (imagen) | +0.0001 | 0.4434 | 1 | +0.000 |  | no |
| unet_pp | T2 Bootstrap pareado | +0.0001 | 0.826 | 1 | +0.000 | [-0.0004, 0.0007] | no |
| unet_pp | T3 Mann-Whitney (semillas) | +0.0001 | 0.8413 | 1 | +0.120 |  | no |
| mednext | T1 Wilcoxon (imagen) | +0.0022 | 0.0001769 | 0.001061 | +0.161 |  | sí |
| mednext | T2 Bootstrap pareado | +0.0022 | 0.0001 | 0.0008 | +0.161 | [0.0010, 0.0036] | sí |
| mednext | T3 Mann-Whitney (semillas) | +0.0022 | 0.2222 | 0.8889 | +0.520 |  | no |

Holm sobre los 9 p-valores. 'gana' = significativo tras Holm, |δ| >= 0.147 y diferencia a favor del original.