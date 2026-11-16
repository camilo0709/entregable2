# Resultados experimentales

Todos los números proceden de `resultados/e3/` (test sellado: 80 positivas + 40 negativas, GT plata SAM2,
techo humano Dice 0.9879). HPO: 6 intentos por modelo; paciencia 15; Holm sobre 9 valores p.

## Benchmark (5 semillas base, Dice test positivas)

| Modelo | Dice |
|---|---|
| attention_unet | 0.5678 |
| unet_pp | 0.5518 |
| mednext | 0.5094 |
| transunet | 0.5048 |
| deeplabv3p | 0.5007 |
| unet | 0.4985 |
| unetr | 0.4763 |
| swin_unet | 0.4758 |
| segformer | 0.4740 |

Friedman χ² = 102.49, p = 1.3e-18. AUC de detección: attention_unet 0.926, unet_pp 0.911.

## Original vs. top-3 (10 semillas)

| Modelo | Dice (10 sem.) | Δ vs LACCA | p Holm | Resultado |
|---|---|---|---|---|
| lacca_unet | 0.5651 ± 0.022 | — | — | — |
| attention_unet | 0.5595 | +0.0056 | 1.0 | empate |
| unet_pp | 0.5574 | +0.0078 | 1.0 | empate |
| mednext | 0.5105 | +0.0546 | ≈0.001 | LACCA gana |

## Ablaciones (5 semillas)

| Variante | Δ Dice vs LACCA completo | p Holm |
|---|---|---|
| eje fijo | −0.0029 | ≥ 0.25 |
| sin asimetría | +0.0008 | ≥ 0.25 |
| atención global | +0.0109 | ≥ 0.25 |

Ninguna componente muestra aporte significativo: **no hay evidencia de que el eje aprendido o el residuo
ayuden**; la ventaja sobre U-Net proviene sobre todo del backbone y del HPO.

## Calibración (semilla 42, ECE antes → después de escalado de temperatura)

LACCA 0.0368 → 0.0306; attention_unet 0.0327 → 0.0210; unet_pp 0.0459 → 0.0473; mednext 0.0349 → 0.0420; unet 0.1038 → 0.1087.

## Robustez (Dice positivas, semilla 42)

| Perturbación | LACCA | Attention | U-Net++ | MedNeXt | U-Net |
|---|---|---|---|---|---|
| limpio | .558 | .586 | .547 | .531 | .514 |
| ruido 0.5 | .146 | .270 | .196 | 0 | .228 |
| desenfoque 2 | .174 | .172 | .211 | .441 | .331 |
| contraste 0.5 | .375 | .181 | .351 | .421 | .278 |
| brillo +0.5 | .504 | .477 | .382 | .473 | .449 |

LACCA es frágil ante ruido y desenfoque. Es una sola semilla: indicativo, no concluyente.

## Validación externa (máscaras pulmonares; re-entrenado con 85 %, 15 % retenido)

| Dataset | LACCA | U-Net++ | MedNeXt | Attention |
|---|---|---|---|---|
| Montgomery (22) | .9704 | .9703 | .9682 | .9675 |
| Shenzhen (86) | .9577 | .9575 | .9571 | .9571 |

Shenzhen: Friedman p = 0.52 (sin diferencias). Montgomery: la marca «gana» solo a nivel imagen (Δ 0.002–0.003 < δ = 0.01);
a nivel semilla no es significativo. No es evaluación zero-shot ni de lesiones.

## Interpretación

LACCA iguala a los mejores baselines y los supera claramente solo sobre MedNeXt. El techo de ~0.57 frente a
0.9879 humano sugiere que el cuello de botella es el GT de plata y el tamaño de las lesiones, no la arquitectura.
