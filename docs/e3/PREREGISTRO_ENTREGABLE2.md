# PREREGISTRO — ENTREGABLE 2: BENCHMARK DE SEGMENTACIÓN TB

- **Estado:** 🔒 preregistrado **ANTES** de cualquier evaluación sobre el test
- **Fecha de sellado:** 2026-10-20
- **Split:** `datos/splits/split_manifest.json` — SHA256
  `9B382D82D24E79C81494C70FD94FD375AC2AE33667CD4DE17D3C04993930EB5F`
  (semilla 42; train 165 / val 29 / test_pos 80 / test_neg 40; `sealed: true`)
- **Regla de oro:** cualquier desviación de este documento debe registrarse en
  `docs/PREREGISTRO_ENMIENDAS.md` con fecha, motivo y **antes** de ver
  resultados del test que la motivan.

---

## 1. Pregunta de investigación y gap

**Pregunta:** bajo un protocolo idéntico y libre de fuga de datos, ¿qué familia
de arquitecturas (CNN, CNN+Transformer, Transformer puro) segmenta mejor las
lesiones de tuberculosis en radiografías de tórax, y las diferencias son
estadísticamente sustantivas (no solo numéricas)?

**Gap declarado:** la literatura compara modelos con protocolos dispares
(diferente preproceso, split, pérdida y umbral), lo que impide una conclusión
de conjunto; además, la mayoría valida contra GT derivado de bounding boxes o
de segmentadores automáticos, con el sesgo circular que eso conlleva. Este
benchmark ataca ambos frentes: protocolo único verificable y test con
anotación humana (Decisión D1).

**Hipótesis contrastable (dirección del gap):**
las diferencias observadas entre familias en estudios previos se reducen o
invierten al estandarizar el protocolo; en particular, no esperamos que las
arquitecturas transformer superen de forma consistente a las CNN en un
dataset de este tamaño (165 imágenes de train).

## 2. Hipótesis

| ID | Tipo | Hipótesis | H₀ |
|---|---|---|---|
| **H1** | Confirmatoria (primaria) | Existen diferencias en **Dice** entre los 10 modelos sobre el test primario | Todas las distribuciones de Dice son iguales (Friedman) |
| **H1.a** | Confirmatoria (post-hoc) | El mejor modelo supera a al menos un modelo con diferencia **> δ = 0,01** en Dice (clínicamente sustantiva) | Δ ≤ δ en todas las comparaciones |
| **H1.b** | Secundaria (exploratoria) | El orden de los modelos se replica al medir con IoU, HD95 y ASSD | Los rankings no difieren de los del Dice |
| **H1.c** | Secundaria (exploratoria) | Los modelos difieren en su capacidad de **detección** (AUC y aciertos de presencia/ausencia en 80 pos + 40 neg) | AUC iguales (DeLong) / tablas discordantes iguales (McNemar) |
| **H2** | Gap / futuro | Un modelo ligero entrenado por destilación de SAM 2 y calibrado iguala al mejor del benchmark con órdenes de magnitud menos parámetros | El modelo destilado es inferior en más de δ |

H2 queda **declarada como dirección de investigación** (Entregable 3); no
participa en la inferencia de este entregable.

## 3. Diseño

| Elemento | Valor |
|---|---|
| Modelos | 10 (U-Net, Attention U-Net, U-Net++, DeepLabV3+, nnU-Net, TransUNet*, Swin-UNet*, UNETR, SegFormer, MedNeXt†) — *sustitutos declarados en `modelos_tb.py`, †SegResNet como par disponible |
| Baselines | empty_mask, bbox_mask (caja TBX11K), unet_fuerte (excluidos de Friedman; en Tabla A y comparaciones exploratorias) |
| Semillas | 5 por modelo: 42, 123, 777, 2024, 31415 (mínimo exigido: 5) |
| Unidad de análisis | imagen |
| Train / Val | 165 / 29, GT = pseudomáscara SAM 2 (etiqueta débil, idéntica para todos) |
| **Test primario** | 80 positivas, GT **manual** (protocolo `PROTOCOLO_ANOTACION.md`, QC: Dice inter-anotador ≥ 0,70) |
| Test secundario | 40 negativos sanos, máscara vacía por construcción |
| Protocolo idéntico | entrada 1 canal 512×512, salida 1 logit; pérdida DiceBCE 0,5/0,5; AdamW 1e-4; cosine; ≤150 épocas, early stopping paciencia 25 sobre Dice de val; umbral 0,5 |
| Fuga de datos | split sellado, comprobado (sin duplicados por MD5/dHash/correlación); el test **no** se toca hasta congelar GT y preregistro |

## 4. Métricas

- **Primaria:** Dice (binarización prob > 0,5).
- **Secundarias:** IoU/mIoU, HD95 (px), ASSD (px).
- **Margen de relevancia:** δ = 0,01 en Dice.
- Definiciones y casos borde (ambas vacías → Dice=1; GT vacío → Dice=0;
  HD95/ASSD NaN si hay máscara vacía) fijadas en `codigo/metricas_tb.py`.
- **Análisis de sensibilidad:** cada modelo se promedia sobre sus 5 semillas
  antes del análisis por pares; se reporta además la desviación entre semillas
  y se repite Friedman una vez por semilla (5 réplicas) para mostrar la
  estabilidad del ranking.

## 5. Tablas de resultados (formato fijado)

### Tabla A — Descriptiva (por modelo)
Una fila por modelo (y baseline): Dice medio ± DE entre semillas (sobre el
promedio por imagen del test primario), IoU, HD95 (con `n` definido),
ASSD, parámetros (M), tiempo GPU (s/época), AUC de detección (120 imágenes),
y las 5 corridas individuales en apéndice.

### Tabla B — Ranking global (Friedman)
Modelo, rango medio de Friedman, χ²_F de Friedman, p, y F_F de Iman-Davenport
con su p. El modelo con mejor rango se declara **mejor del benchmark**.

### Tabla C — Comparaciones por pares vs. el mejor modelo
Para cada uno de los 9 modelos: Δ Dice medio (modelo − mejor), IC 95 % BCa
(≥ 2000 re-muestreos a nivel de imagen), p de Wilcoxon con **Holm**, tamaño
del efecto r = Z/√N, y decisión: **mejor / equivalente (|Δ| < δ) / peor**.

## 6. Plan de inferencia formal

1. **Omnibus:** Friedman sobre la matriz 10 × 80 (modelos × imágenes, valores
   = Dice medio de las 5 semillas). Corrección de Iman-Davenport:
   `F_F = (n−1)χ²_F / (n(k−1) − χ²_F)`, α = 0,05.
2. **Post-hoc:** Wilcoxon con signo (modelo vs mejor, por imagen), 9
   comparaciones, **Holm-Bonferroni**, α = 0,05. Tamaños de efecto r.
3. **Equivalencia (δ):** TOST implícito vía IC 95 % BCa (2000 re-muestreos
   estratificados por imagen): equivalente si el IC completo cae en
   [−0,01; +0,01]. Un modelo no significativamente peor **no** se declara
   equivalente sin este paso.
4. **DeLong (H1.c):** tarea de detección en las 120 imágenes (80 pos + 40
   neg). Score continuo predefinido: **masa de probabilidad predicha**
   (suma de sigmoide/1000). Comparación pareada de AUC m vs mejor, Holm.
5. **McNemar exacto (H1.c):** decisión binaria por imagen (¿algún píxel > 0,5?)
   contra presencia real de lesión; 9 comparaciones, Holm.
6. **Bootstrap:** ≥ 2000 re-muestreos con reemplazo a nivel de imagen para
   todos los IC reportados (BCa).
7. **Multiplicidad:** solo Dice es confirmatorio; IoU/HD95/ASSD y subgrupos
   (n_boxes 1 vs ≥ 2; los 39 vs 41 del test) son **exploratorios** y se
   etiquetan como tales.
8. **Datos faltantes:** sin imputación. HD95/ASSD undefined se excluye del
   promedio **declarando el denominador** en Tabla A. Cero outliers removidos
   por definición.

## 7. Blindaje contra el sesgo post-hoc

1. El GT del test se congela (`validar_ground_truth_test.py` → exit 0 +
   `ANOTACION_METRICAS.json`) **antes** de la primera inferencia sobre test.
2. Ningún modelo se re-entrena observando resultados de test: la selección de
   época usa **solo** val (Dice de val), y el umbral queda fijado en 0,5.
3. Los sustitutos declarados (TransUNet, Swin-UNet, MedNeXt→SegResNet) se
   mantienen aunque sus resultados sean malos; cambiarlos exigiría enmienda.
4. El preregistro y el manifiesto de split se suben a Drive con fecha; el
   hash de §cabecera prueba la integridad.

## 8. Presupuesto y cronograma (Colab Pro, GPU L4)

| Bloque | Contenido | CU aprox. |
|---|---|---|
| 1 | U-Net, Attention U-Net, U-Net++ × 5 semillas | ~9 |
| 2 | DeepLabV3+, SegFormer, TransUNet, Swin-UNet × 5 | ~12 |
| 3 | UNETR, MedNeXt, unet_fuerte × 5 | ~9 |
| 4 | nnU-Net (pipeline propio) × 5 | ~8 |
| Reserva | reintentos, sensibilidad, bootstrap local | resto de ~100 |

Inferencia estadística: **local** (CPU, sin costo de CU). Estimación
conservadora: benchmark completo ≈ 38–45 CU de los ~100 disponibles.

## 9. Limitaciones declaradas de antemano

1. GT de train/val = pseudomáscaras SAM 2 (etiqueta débil): puede favorecer
   modelos con sesgos afines a SAM 2; el **test manual** mitiga el efecto en
   la conclusión.
2. Sin ID de paciente en TBX11K: no se puede excluir solapamiento de pacientes
   entre splits (se verificó ausencia de duplicados de imagen).
3. n = 80 positivas: poder moderado para diferencias pequeñas; δ = 0,01 se
   declara por relevancia clínica, no por potencia.
4. Sustitutos de MedNeXt/TransUNet/Swin-UNet: se comparan implementaciones
   disponibles por pip, no los papers originales.

---

*Vendrá firma del responsable al aprobar: ______________________ (fecha: ____)*
