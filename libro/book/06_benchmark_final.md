# Sección 6 · Benchmark final: inferencia estadística

## 6.1 Qué mide este análisis 

Los valores de `resumen_runs.csv` están en la columna **`dice_val`**: son Dice de **validación**, la métrica con la que se elige la época (early stopping). **No son resultados de test.** Por tanto:

- Este análisis es **de validación** y no sustituye la inferencia confirmatoria del preregistro.
- La inferencia confirmatoria (H1, H1.a) se hará sobre el test primario (80 positivas, GT de plata según la decisión D1-R1) una vez cerrado el GT y ejecutado el test.

## 6.2 Diseño tal como se preregistró

Fuente: `PREREGISTRO_ENTREGABLE2.md` y `DECISION_D1_GROUND_TRUTH.md` (R1).

- **Unidad de análisis:** imagen (80 positivas en test). Friedman sobre la matriz 10 × 80.
- **Post-hoc:** Wilcoxon con signo por imagen, Holm sobre 9 comparaciones contra el mejor modelo.
- **Equivalencia:** IC 95 % BCa con δ = 0.01 en Dice.
- **Detección (H1.c):** AUC con DeLong y McNemar exacto sobre las 120 imágenes (80 pos + 40 neg).
- **Multiplicidad:** solo Dice es confirmatorio; IoU, HD95, ASSD y subgrupos son exploratorios.
- **GT del test (R1):** plata (pseudomáscara SAM 2), con validación humana de una submuestra de 20 imágenes que fija un techo de desempeño. Este cambio se aprobó antes de entrenar y está documentado.

## 6.3 Modelos y exclusiones

| Modelo | Estado | Nota |
|---|---|---|
| U-Net | Completado | |
| U-Net++ | Completado | |
| DeepLabV3+ | Completado | |
| TransUNet | Completado | Sustituto declarado (implementación pip) |
| Swin-UNet | Completado | Sustituto declarado (implementación pip) |
| UNETR | Completado | |
| SegFormer | Completado | |
| MedNeXt | Completado | Sustituto declarado (SegResNet disponible como par) |
| Attention U-Net | Completado, **sospechoso** | Valores idénticos a U-Net en las 5 semillas. Verificar antes de usar |
| unet_fuerte | Baseline | Excluido de Friedman según preregistro. Idéntico a U-Net en las 5 semillas |

Attention U-Net y unet_fuerte quedan fuera del análisis de validación por la misma razón.

## 6.4 Resultados de validación (8 modelos × 5 semillas)

| Modelo | Media Dice_val | DE | IC 95 % (bootstrap sobre semillas) | Rango medio |
|---|---|---|---|---|
| DeepLabV3+ | 0.5061 | 0.0118 | [0.4969, 0.5150] | 1.80 |
| U-Net++ | 0.5016 | 0.0159 | [0.4885, 0.5142] | 2.20 |
| TransUNet | 0.4979 | 0.0256 | [0.4793, 0.5171] | 2.80 |
| U-Net | 0.4846 | 0.0178 | [0.4706, 0.4985] | 4.40 |
| SegFormer | 0.4725 | 0.0127 | [0.4621, 0.4830] | 5.00 |
| MedNeXt | 0.4670 | 0.0242 | [0.4500, 0.4886] | 5.60 |
| UNETR | 0.4506 | 0.0159 | [0.4367, 0.4611] | 6.80 |
| Swin-UNet | 0.4527 | 0.0137 | [0.4429, 0.4644] | 7.40 |

Advertencia: el IC y los tests de abajo usan las **5 semillas** como unidad, no las imágenes. Eso es una **sensibilidad a semillas**, que el preregistro pide como análisis secundario. No es la inferencia primaria.

**Supuestos (Shapiro–Wilk por modelo, Levene):** no se rechazan. Con n = 5 la potencia es baja.

**Friedman (bloques = semillas):** χ² = 25.53, p = 0.0006. Iman–Davenport F = 10.79, p < 0.001.

**Nemenyi (CD = 4.70):** DeepLabV3+ supera a Swin-UNet y UNETR. Ningún otro par supera el CD.

**Wilcoxon pareado + Holm:** ningún par es significativo. Con 5 bloques, el p mínimo bilateral es 0.0625, así que el test no puede detectar diferencias con este n.

**Tamaño de efecto (d_z) vs DeepLabV3+:** U-Net 0.99, TransUNet 0.39, U-Net++ 0.32, MedNeXt 2.12, SegFormer 2.39, UNETR 2.61, Swin-UNet 4.66.

## 6.5 Lo que falta para cerrar la sección

1. **Dice de test** por imagen, por modelo (80 pos) y especificidad en test_neg (40 sanos). No están en el Drive.
2. Friedman, Wilcoxon + Holm, BCa, δ-equivalencia, DeLong y McNemar sobre esos datos, según el plan del preregistro.
3. Dice inter-anotador / techo humano (submuestra de 20 imágenes).
4. Verificar que Attention U-Net y unet_fuerte no sean copias de U-Net.

## 6.6 Conclusión provisional (solo validación)

En validación, DeepLabV3+ tiene el mejor Dice medio y rango medio, y supera de forma significativa a Swin-UNet y UNETR bajo Nemenyi. Las diferencias con U-Net++ y TransUNet no son concluyentes con 5 semillas. **Esto no es una conclusión sobre test.**
