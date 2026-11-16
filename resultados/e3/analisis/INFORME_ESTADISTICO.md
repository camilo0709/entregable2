# Informe estadístico — Entregable 2

- **Generado:** 2026-11-16T03:03:03
- **Test primario:** 80 imágenes (GT plata D1-R1)
- **Test secundario:** 40 negativas (máscara vacía)
- **Modelos en Friedman:** 9 (unet, attention_unet, unet_pp, deeplabv3p, transunet, swin_unet, unetr, segformer, mednext)
- **Mejor modelo:** **attention_unet** (menor rango medio)

- **Excluidos por el usuario:** nnunet

## Tabla B — Friedman

χ²_F = 102.4881, p = 1.32e-18; F_F (Iman-Davenport) = 15.0630, p = 2.59e-20

| Puesto | Modelo | Rango medio | Dice medio |
|---:|---|---:|---:|
| 1 | attention_unet | 3.000 | 0.5678 |
| 2 | unet_pp | 3.481 | 0.5518 |
| 3 | mednext | 4.888 | 0.5094 |
| 4 | transunet | 5.044 | 0.5048 |
| 5 | deeplabv3p | 5.256 | 0.5007 |
| 6 | unet | 5.381 | 0.4985 |
| 7 | swin_unet | 5.719 | 0.4758 |
| 8 | unetr | 6.106 | 0.4763 |
| 9 | segformer | 6.125 | 0.4740 |

## Tabla C — Comparaciones vs el mejor

| Modelo | Δ Dice | IC 95% BCa | p Wilcoxon | p Holm | r | Decisión |
|---|---:|---|---:|---:|---:|---|
| unet | -0.0694 | [-0.0898, -0.0472] | 0 | 0 | -0.598 | peor |
| unet_pp | -0.0160 | [-0.0327, +0.0025] | 0.0222 | 0.0222 | -0.256 | peor |
| deeplabv3p | -0.0671 | [-0.0872, -0.0468] | 0 | 0 | -0.602 | peor |
| transunet | -0.0631 | [-0.0809, -0.0460] | 0 | 0 | -0.643 | peor |
| swin_unet | -0.0921 | [-0.1180, -0.0656] | 0 | 0 | -0.644 | peor |
| unetr | -0.0916 | [-0.1211, -0.0593] | 0 | 0 | -0.587 | peor |
| segformer | -0.0938 | [-0.1208, -0.0657] | 0 | 0 | -0.609 | peor |
| mednext | -0.0584 | [-0.0817, -0.0373] | 6e-06 | 1.2e-05 | -0.506 | peor |

## Tabla A — Descriptiva

| Modelo | Dice test | IoU | HD95 (n) | ASSD (n) | Params (M) | s/época | Dice val | AUC det. | Nota |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unet | 0.4985 ± 0.0142 | 0.3638 | 150.54 (80) | 47.83 (80) | 24.43 | 11.1 | 0.4840 | 0.7839 |  |
| attention_unet | 0.5678 ± 0.0209 | 0.4314 | 120.78 (80) | 35.47 (80) | 24.54 | 8.3 | 0.5694 | 0.9258 | re-entrenado tras corregir el gate (E1) |
| unet_pp | 0.5518 ± 0.0129 | 0.4202 | 125.35 (79) | 37.45 (79) | 26.07 | 4.1 | 0.5668 | 0.9109 |  |
| deeplabv3p | 0.5007 ± 0.0233 | 0.3646 | 142.07 (79) | 43.63 (79) | 26.67 | 2.7 | 0.5019 | 0.8301 |  |
| transunet | 0.5048 ± 0.0230 | 0.3690 | 136.79 (80) | 41.33 (80) | 33.81 | 3.3 | 0.4996 | 0.8449 | sustituto declarado: smp.Unet resnet50 + scse |
| swin_unet | 0.4758 ± 0.0154 | 0.3456 | 142.02 (79) | 44.35 (79) | 27.47 | 4.5 | 0.5007 | 0.7508 | sustituto declarado: smp.Unet mit_b2 |
| unetr | 0.4763 ± 0.0218 | 0.3373 | 184.65 (80) | 61.49 (80) | 87.92 | 11.0 | 0.4615 | 0.6864 |  |
| segformer | 0.4740 ± 0.0238 | 0.3376 | 148.31 (80) | 45.11 (80) | 24.72 | 8.5 | 0.4656 | 0.7069 |  |
| mednext | 0.5094 ± 0.0315 | 0.3734 | 134.45 (79) | 40.87 (79) | 6.30 | 10.8 | 0.5417 | 0.7999 | sustituto declarado: MONAI SegResNet |
| unet_fuerte | 0.5236 ± 0.0175 | 0.3889 | 142.97 (80) | 44.12 (80) | 24.43 | 10.1 | 0.5192 | 0.7789 | duplicado de unet (E4); excluido del Friedman |
| empty_mask | 0.0000 ± 0.0000 | 0.0000 | — (0) | — (0) | — | — | — | — | baseline sin red (cota inferior) |
| bbox_mask | 0.0000 ± 0.0000 | 0.0000 | — (0) | — (0) | — | — | — | — | baseline sin red (cajas TBX11K) |

## H1.c — Detección (DeLong / McNemar, Holm)

| Prueba | Modelo | vs | AUC / aciertos | z | p | p Holm |
|---|---|---|---:|---:|---:|---:|
| DeLong | unet | attention_unet | 0.832 vs 0.972 | -8.411 | 0 | 0 |
| DeLong | unet_pp | attention_unet | 0.955 vs 0.972 | -1.367 | 0.172 | 0.172 |
| DeLong | deeplabv3p | attention_unet | 0.903 vs 0.972 | -4.558 | 5e-06 | 1.5e-05 |
| DeLong | transunet | attention_unet | 0.898 vs 0.972 | -4.383 | 1.2e-05 | 2.3e-05 |
| DeLong | swin_unet | attention_unet | 0.808 vs 0.972 | -8.994 | 0 | 0 |
| DeLong | unetr | attention_unet | 0.720 vs 0.972 | -16.231 | 0 | 0 |
| DeLong | segformer | attention_unet | 0.730 vs 0.972 | -13.605 | 0 | 0 |
| DeLong | mednext | attention_unet | 0.853 vs 0.972 | -6.876 | 0 | 0 |
| McNemar | unet | attention_unet | 80 vs 81 (n10=0, n01=1) | — | 1 | 1 |
| McNemar | unet_pp | attention_unet | 83 vs 81 (n10=3, n01=1) | — | 0.625 | 1 |
| McNemar | deeplabv3p | attention_unet | 79 vs 81 (n10=0, n01=2) | — | 0.5 | 1 |
| McNemar | transunet | attention_unet | 80 vs 81 (n10=0, n01=1) | — | 1 | 1 |
| McNemar | swin_unet | attention_unet | 80 vs 81 (n10=0, n01=1) | — | 1 | 1 |
| McNemar | unetr | attention_unet | 80 vs 81 (n10=0, n01=1) | — | 1 | 1 |
| McNemar | segformer | attention_unet | 80 vs 81 (n10=0, n01=1) | — | 1 | 1 |
| McNemar | mednext | attention_unet | 79 vs 81 (n10=0, n01=2) | — | 0.5 | 1 |

## Sensibilidad — Friedman por semilla

| Semilla | χ² | p | Orden |
|---:|---:|---:|---|
| 42 | 84.0547 | 0 | attention_unet > unet_pp > mednext > transunet > deeplabv3p > unet > swin_unet > segformer > unetr |
| 123 | 62.0841 | 0 | unet_pp > attention_unet > mednext > segformer > unet > unetr > transunet > deeplabv3p > swin_unet |
| 777 | 63.5756 | 0 | attention_unet > unet_pp > transunet > deeplabv3p > swin_unet > mednext > unet > unetr > segformer |
| 2024 | 60.5912 | 0 | attention_unet > unet_pp > transunet > unet > mednext > swin_unet > deeplabv3p > unetr > segformer |
| 31415 | 55.6674 | 0 | attention_unet > unet_pp > deeplabv3p > mednext > unet > transunet > swin_unet > segformer > unetr |
