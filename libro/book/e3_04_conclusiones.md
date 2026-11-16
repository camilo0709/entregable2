# Conclusiones, limitaciones y declaración de uso de IA

## Veredicto

| Criterio | Estado |
|---|---|
| R1 (implementación original y reproducible) | Cumple |
| R2 (evaluación comparada con protocolo estadístico) | Cumple |
| R3 (superioridad significativa sobre mejor baseline) | **No cumple** |

Con R1 y R2 y sin R3, la nota máxima del Cuadro 7 es **4.0**.

## Limitaciones declaradas

- `lacca_global` corrió en el lote 4 (no simultáneo con el resto).
- HPO con paciencia mixta; posible sesgo de selección del top-3 por validación.
- Calibración, robustez y Grad-CAM con una sola semilla.
- Validación externa con máscaras pulmonares, re-entrenada; no mide lesiones.
- GT de plata (SAM2): el techo del Dice está acotado por la calidad del GT.
- nnU-Net no se evaluó (limitación declarada).
- Discrepancia de umbral: el preregistro usa δ_Dice = 0.01 en el análisis; la nota de integración menciona 0.02.
- Búsqueda bibliográfica sin Scopus/IEEE.

## Trabajo futuro

GT experto, más semillas en robustez, eje fijo como línea base principal, estudio en datos con asimetría real.

## Declaración de autoría y uso de herramientas de IA

Autores: **Mujica Camilo y Romero Carlos**. Se usaron asistentes de IA (Claude, Anthropic) como apoyo en
programación, análisis y redacción. Los autores revisaron y son responsables del contenido. La declaración
firmada la completan los autores.
