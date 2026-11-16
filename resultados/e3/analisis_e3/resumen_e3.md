# Comparaciones del modelo original (E3)

Original: `lacca_unet`; rivales: attention_unet, unet_pp, mednext. Semillas por modelo: {'lacca_unet': 10, 'attention_unet': 10, 'unet_pp': 10, 'mednext': 10}.

Friedman (imágenes): chi2=22.042, p=6.395e-05

| rival | prueba | Δ Dice | p | p Holm | Cliff δ | IC 95 % | gana |
|---|---|---|---|---|---|---|---|
| attention_unet | T1 Wilcoxon (imagen) | +0.0056 | 0.601 | 1 | +0.008 |  | no |
| attention_unet | T2 Bootstrap pareado | +0.0056 | 0.4908 | 1 | +0.008 | [-0.0104, 0.0216] | no |
| attention_unet | T3 Mann-Whitney (semillas) | +0.0056 | 0.6232 | 1 | +0.140 |  | no |
| unet_pp | T1 Wilcoxon (imagen) | +0.0078 | 0.4144 | 1 | +0.018 |  | no |
| unet_pp | T2 Bootstrap pareado | +0.0078 | 0.3314 | 1 | +0.018 | [-0.0077, 0.0233] | no |
| unet_pp | T3 Mann-Whitney (semillas) | +0.0078 | 0.6232 | 1 | +0.140 |  | no |
| mednext | T1 Wilcoxon (imagen) | +0.0546 | 0.0009348 | 0.006544 | +0.163 |  | sí |
| mednext | T2 Bootstrap pareado | +0.0546 | 0.0001 | 0.0009 | +0.163 | [0.0269, 0.0825] | sí |
| mednext | T3 Mann-Whitney (semillas) | +0.0546 | 0.0002461 | 0.001969 | +0.980 |  | sí |

Holm sobre los 9 p-valores. 'gana' = significativo tras Holm, |δ| >= 0.147 y diferencia a favor del original.