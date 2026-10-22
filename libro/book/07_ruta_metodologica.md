# Sección 7 · Ruta metodológica según el tipo de datos (I1–I8)

## 7.1 Contexto del problema

Imagen 2D monocroma (radiografía de tórax, 512×512), baja señal (lesiones con contraste de unos 10–30 niveles sobre 0–255), GT de plata (pseudomáscara de SAM 2 sobre la caja de TBX11K) y N moderado (194 imágenes de pool de entrenamiento, 80 positivas y 40 negativas en test). Estas condiciones determinan la ruta de abajo.

## 7.2 Ficha de ruta I1–I8

| Paso | Decisión tomada | Justificación para este problema | Alternativas descartadas |
|---|---|---|---|
| **I1. Tarea y unidad de independencia** | Segmentación semántica binaria de lesiones TB. Unidad de análisis: **imagen**. | TBX11K da cajas, no máscaras, pero el objetivo clínico es píxel a píxel. La unidad por paciente no es posible porque TBX11K no tiene ID de paciente. | Clasificación o detección (no responden la pregunta de segmentación). Unidad paciente (no verificable). |
| **I2. Duplicados y particionado** | Hashing perceptual (dHash) y MD5 entre splits; partición fija con semilla 42 y sellada. | Resultado: 0 duplicados exactos, 48 casi-duplicados sin correlación >0.99, 0 IDs repetidos entre splits. Limitación declarada: sin ID de paciente. | Partición sin comprobación (riesgo de fuga). |
| **I3. Normalización y resolución** | Resolución fija 512×512 y 1 canal. Media y DE por canal calculadas **solo en train**. | La imagen es monocroma y ya está a 512×512, así que el padding no aplica. | Estadísticas de todo el dataset (fuga). *Pendiente de verificar: valores de media y DE.* |
| **I4. Aumentos de datos** | Aumentos aplicados solo a train, con transformación sincronizada de imagen y máscara. | En segmentación, un aumento que desalinea imagen y máscara invalida la etiqueta. | Aumentos en val o test. *Pendiente de verificar: lista exacta de aumentos.* |
| **I5. Baselines y modelos** | Baselines: máscara vacía, caja TBX11K (`bbox_mask`) y `unet_fuerte`. Los 10 modelos con el mismo protocolo: 1 canal, 512×512, pérdida DiceBCE 0.5/0.5, AdamW 1e-4, coseno, ≤150 épocas, early stopping con paciencia 25 sobre Dice de validación, umbral 0.5. | Presupuesto y entrada idénticos, como exige el preregistro. | Protocolos distintos por modelo (comparación no justa). |
| **I6. Política de inferencia** | Sin test-time augmentation, para todos los modelos. | Aplicar TTA solo a algunos modelos sesgaría la comparación. *Pendiente de verificar: confirmar que no se usó TTA en ningún modelo.* | TTA en un subconjunto de modelos. |
| **I7. Pruebas y bootstrap** | Friedman sobre 10 × 80 imágenes; Wilcoxon con signo y Holm; BCa con ≥2000 remuestreos a nivel de imagen. | El agrupamiento por paciente no es posible. **Discrepancia:** el ítem I7 pide bootstrap agrupado por paciente, mientras el preregistro usa remuestreo por imagen. Hay que declararla. | Bootstrap por paciente (datos no disponibles). |
| **I8. Interpretabilidad y errores** | Grad-CAM sobre los 3 mejores modelos, con solapamiento frente a la caja y la máscara GT, comparado con un baseline geométrico. Análisis por subgrupos (número de cajas, tipo de TB, calidad de imagen) y control OOD con `test_neg` (falsos positivos). | Detecta atajos (bordes, marcas, texto incrustado) en radiografías. | *Attention rollout*: no está en el plan. |

## 7.3 Justificación general de la ruta

- **Segmentación con U-Net como base y modelos alternativos** responde a la pregunta del preregistro (qué familia segmenta mejor).
- **GT de plata** es la única opción con máscara disponible sin anotación experta. Se mitiga con la submuestra humana de 20 imágenes y se declara como limitación.
- **Friedman + Nemenyi/Wilcoxon** es el enfoque estándar para comparar varios modelos sobre varios bloques con n moderado.

## 7.4 Tablas A/B/C

Se refieren desde la Sección 6:
- **Tabla A:** descriptiva por modelo (Dice medio ± DE entre semillas, IoU, HD95, ASSD, parámetros, tiempo por época, AUC de detección).
- **Tabla B:** ranking global de Friedman (rango medio, χ², p, Iman–Davenport).
- **Tabla C:** comparaciones contra el mejor modelo (Δ Dice, IC 95 % BCa, p con Holm, r, decisión).

## 7.5 Preregistro Paso 1: qué fue preregistrado y qué fue post-hoc

**Preregistrado** (`PREREGISTRO_ENTREGABLE2.md`, sellado el 2026-10-20):
- Hipótesis H1, H1.a, H1.b, H1.c y H2 (H2 como dirección de investigación, fuera de este entregable).
- Diseño: 10 modelos, 5 semillas (42, 123, 777, 2024, 31415), split sellado con semilla 42 (165 / 29 / 80 / 40 / 64 de reserva).
- Métricas: Dice como primaria; IoU, HD95 y ASSD como secundarias; δ = 0.01.
- Plan de inferencia y blindaje contra sesgo post-hoc.

**Cambios posteriores (post-hoc o enmiendas):**

1. **GT del test de manual a plata** (D1-R1, aprobado el 2026-10-20, antes de fijar la fecha de sellado del preregistro y antes de entrenar). **Conflicto:** el preregistro todavía describe GT manual. El documento debe declarar que la enmienda prevalece.
2. **nnU-Net no disponible** (`no_disponible` en las 5 semillas). Se sale de la comparación estadística y pasa a la comparación de Pareto de la Sección 7/8.
3. **Sustitutos declarados** (TransUNet, Swin-UNet, MedNeXt → SegResNet). Estaban en el preregistro, no son post-hoc, pero conviene recordar que no son los papers originales.
4. **Attention U-Net y unet_fuerte idénticos a U-Net** en Dice de validación en las 5 semillas. Es un hallazgo posterior al preregistro. Hasta verificarlo, se excluyen del análisis.
5. **Discrepancia del hash del split:** el `split_manifest.json` v1 no coincide con los CSV recuperados. Se creó v2 como línea base vigente, sellada antes de entrenar. Declarado en el Data Card.

## 7.6 Limitaciones que se declaran

- Sin ID de paciente en TBX11K: no se puede excluir solapamiento entre pacientes.
- GT de train/val = pseudomáscaras de SAM 2 (etiqueta débil), con posible sesgo hacia SAM 2.
- n = 80 positivas en test: poder moderado para diferencias pequeñas.
- Sustitutos de MedNeXt, TransUNet y Swin-UNet, no los papers originales.
