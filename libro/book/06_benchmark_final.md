# §6 · Benchmark final: inferencia estadística

> **Estado:** en espera de resultados (20 % de la rúbrica).

Contendrá (script `codigo/estadistica_tb.py`, en construcción):

- **Normalidad y homocedasticidad:** Shapiro–Wilk y Levene por modelo.
- **Comparación global:** Friedman (+ Iman–Davenport) sobre Dice de test.
- **Post-hoc:** Nemenyi con CD, y Wilcoxon pareado + Holm (corrección múltiple).
- **Efecto e incertidumbre:** tamaño de efecto y bootstrap IC 95 %.
- **Robustez:** dispersión entre semillas, análisis por estratos (n_boxes, tipo),
  control OOD `test_neg` (falsos positivos).
- **Grad-CAM:** top-3 modelos, solapamiento con bbox/máscara GT vs baseline geométrico.
- **Ablación:** ≥3 factores (p. ej., pérdida BCE+Dice, resolución, aumentos).
- **Análisis Pareto:** Dice vs coste (parámetros/tiempo) — incluye la
  auto-configuración de nnU-Net (`docs/NNUNET_PIPELINE.md` §7).
- **Tablas A/B/C** y figuras del §7 del informe.
