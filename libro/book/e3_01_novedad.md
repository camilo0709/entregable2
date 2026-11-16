# Novedad y estado del arte actualizado

## Búsqueda documentada (16-nov-2026)

| # | Fuente | Consulta | Hallazgo relevante |
|---|---|---|---|
| 1 | Web (buscador general) | learned symmetry axis contralateral cross-attention chest X-ray lesion segmentation | CE-Net (contralateral, eje = línea espinal, detección) |
| 2 | Web | tuberculosis lesion segmentation chest radiograph bilateral symmetry attention TBX11K | SymFormer / SymAttention (TBX11K, detección y clasificación) |
| 3 | arXiv | mirrored feature map cross-attention symmetry medical image segmentation learnable reflection | SACA (cerebro 3D), MFASSL (SSL, eje estimado), SATNet |
| 4 | PubMed/PMC | asymmetry-aware network chest X-ray tuberculosis segmentation contralateral lung | Solo segmentación pulmonar/TB con U-Net; sin módulo contralateral |
| 5 | Web (español) | atención cruzada contralateral simetría torácica segmentación tuberculosis | Sin resultados específicos |
| 6 | GitHub/HF | symmetry attention chest x-ray segmentation flip feature map | Sin implementaciones equivalentes |

**Limitación declarada:** no se consultaron Scopus ni IEEE Xplore (requieren acceso institucional); los conteos
PRISMA de esas bases deben completarlos los autores. No se afirma ausencia total de trabajos previos.

## Trabajos más cercanos y diferencia

- **CE-Net** (Contralaterally Enhanced Networks, arXiv 2010.04483): usa la línea espinal como eje y un STN
  para extraer parches contralaterales en *detección*. LACCA: eje **aprendido** end-to-end, atención local
  sobre el reflejo en *segmentación*, residuo de asimetría enmascarado.
- **SymFormer / SymAttention + SPE** (Liu et al., TPAMI 2024, arXiv 2307.02848): simetría bilateral en TBX11K para
  *detección y clasificación* con cajas. LACCA opera sobre máscaras densas y garantiza equivarianza al flip.
- **SACA** (arXiv 2407.08948): atención cruzada entre hemisferios en MRI 3D con preentrenamiento.
- **MFASSL** (arXiv 2607.00850): atención espejo en auto-supervisión con eje estimado.

**Posicionamiento:** la novedad es la *combinación* (eje antisimetrizado aprendido + atención local contra el
espejo + residuo de asimetría + compuerta ReZero) en segmentación de lesiones de TB. No se reclama primacía
absoluta; la evidencia empírica (esta sección en adelante) no muestra superioridad de rendimiento.
