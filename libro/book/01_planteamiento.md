# §1 · Definición del problema, preguntas de investigación y dataset

> **Estado:** en preparación (15 % de la rúbrica).

Contendrá:

- **Problema clínico-automatizado:** carga global de TB, escasez de radiólogos,
  segmentación de lesiones como apoyo al tamizaje.
- **Preguntas de investigación RQ1–RQ4** (selladas, ver [índice](index.md)) y
  hipótesis con margen δ de relevancia práctica.
- **Dataset TBX11K:** composición, adquisiciones, limitaciones (→ EDA §2).
- **Ground truth plata D1-R1:** generación con SAM 2, congelamiento, techo
  humano 0.9879, membresía por CSV (§R1.8).
- **Amenazas a la validez** declaradas desde el inicio (preregistro post-hoc,
  sesgo de adquisición AUC 0.842).

Fuentes: `docs/DECISION_D1_GROUND_TRUTH.md`, `datos/splits/split_manifest_v2.json`,
`datos/ground_truth_test/ANOTACION_METRICAS.json`.
