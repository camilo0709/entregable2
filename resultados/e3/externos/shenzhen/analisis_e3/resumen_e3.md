# Comparaciones del modelo original (E3)

Original: `lacca_unet`; rivales: attention_unet, unet_pp, mednext. Semillas por modelo: {'lacca_unet': 5, 'attention_unet': 5, 'unet_pp': 5, 'mednext': 5}.

Friedman (imágenes): chi2=2.260, p=0.5201

| rival | prueba | Δ Dice | p | p Holm | Cliff δ | IC 95 % | gana |
|---|---|---|---|---|---|---|---|
| attention_unet | T1 Wilcoxon (imagen) | +0.0006 | 0.138 | 1 | +0.012 |  | no |
| attention_unet | T2 Bootstrap pareado | +0.0006 | 0.0634 | 0.5706 | +0.012 | [-0.0000, 0.0012] | no |
| attention_unet | T3 Mann-Whitney (semillas) | +0.0006 | 0.4206 | 1 | +0.360 |  | no |
| unet_pp | T1 Wilcoxon (imagen) | +0.0001 | 0.3523 | 1 | +0.008 |  | no |
| unet_pp | T2 Bootstrap pareado | +0.0001 | 0.5804 | 1 | +0.008 | [-0.0004, 0.0006] | no |
| unet_pp | T3 Mann-Whitney (semillas) | +0.0001 | 0.6905 | 1 | +0.200 |  | no |
| mednext | T1 Wilcoxon (imagen) | +0.0006 | 0.3739 | 1 | +0.010 |  | no |
| mednext | T2 Bootstrap pareado | +0.0006 | 0.139 | 1 | +0.010 | [-0.0002, 0.0014] | no |
| mednext | T3 Mann-Whitney (semillas) | +0.0006 | 0.2222 | 1 | +0.520 |  | no |

Holm sobre los 9 p-valores. 'gana' = significativo tras Holm, |δ| >= 0.147 y diferencia a favor del original.