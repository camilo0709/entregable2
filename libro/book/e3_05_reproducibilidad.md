# Reproducibilidad y bibliografía

- Código: `codigo/e3/` (módulo `original_model.py`, pruebas en `codigo/e3/tests/`).
- Cuadernos Colab: `codigo/e3/notebooks/colab_01..05`.
- Resultados: `resultados/e3/` (tabla maestra `experiments/tabla_maestra.csv`).
- Protocolo: `docs/e3/protocolo.md`, enmiendas `docs/e3/PREREGISTRO_ENMIENDAS.md`, hash en `resultados/e3/protocolo.sha256`.
- **Protocolo y etiqueta:** el protocolo está fijado con el tag `protocolo-v1` (commit `c0b7e96`, 15-nov-2026). Ese
  commit es posterior a la ejecución de los entrenamientos: el protocolo no estuvo publicado en el repositorio antes de
  entrenar. La evidencia de lo que se decidió y cuándo es el hash SHA-256 (`resultados/e3/protocolo.sha256`) y las
  enmiendas E1–E7 (`docs/e3/PREREGISTRO_ENMIENDAS.md`). Por ello el análisis se presenta con esta limitación y no como
  un preregistro estricto.
- Semilla global documentada; semillas base 42, 123, 777, 2024, 31415 y extra 7, 11, 99, 1234, 9999.
- ONNX: metadatos y verificación de ida y vuelta en `artifacts/` (los `.onnx` pesados no se versionan).

## Bibliografía

- Liu et al. (2024). Revisiting Computer-Aided Tuberculosis Diagnosis. *IEEE TPAMI*. doi:10.1109/tpami.2023.3330825
- Contralaterally Enhanced Networks for Thoracic Disease Detection (2020). arXiv:2010.04483.
- Symmetry Awareness Encoded Deep Learning Framework for Brain Imaging Analysis (2024). arXiv:2407.08948.
- Mirror-Fusion Attention for Reflection-Aware Self-Supervised Representation Learning. arXiv:2607.00850.
- Rajaraman et al. (2021). Improved Semantic Segmentation of TB-consistent findings in CXRs. *Diagnostics*.

## Declaración de autoría y uso de herramientas

**Autores:** Mujica Camilo y Romero Carlos. Ninguna otra persona participó como autora.

### Reparto del trabajo

Los dos autores se dividieron el **entrenamiento de los modelos** del benchmark y el **registro de sus resultados**
(carpetas de resultados, `resumen_runs.csv` y `por_imagen_test.csv` de cada corrida):

| Autor | Modelos entrenados y registrados |
|---|---|
| Mujica Camilo | SegFormer, U-Net, TransUNet, U-Net++ y DeepLabV3+ |
| Romero Carlos | Los demás: Attention U-Net, MedNeXt, UNETR, Swin-UNet, LACCA-UNet y sus ablaciones (`lacca_eje_fijo`, `lacca_sin_asimetria`, `lacca_global`), además de `unet_fuerte` |

Cada autor ejecutó en Colab los cuadernos de los modelos que le correspondieron y dejó sus archivos en la carpeta
compartida del proyecto.

### Herramientas utilizadas

| Herramienta | Uso | Partes del trabajo |
|---|---|---|
| Claude (Anthropic), asistente de IA | Apoyo en programación, análisis estadístico, depuración (p. ej., error de Grad-CAM en GPU), redacción de capítulos, documentación y organización del repositorio | Código de `codigo/e3/`, análisis de `resultados/e3/`, capítulos `e3_00` a `e3_05`, ficha del modelo, declaración de datos |
| Google Colab (GPU) | Entrenamiento y evaluación de modelos | Cuadernos `colab_01` a `colab_05` |
| SAM 2 (Meta) | Generación del *ground truth* de plata para segmentación | Máscaras de TBX11K (manifiesto con SHA-256) |
| PyTorch, MONAI, smp, timm, SciPy | Modelos, métricas y pruebas estadísticas | Código y análisis |
| Búsqueda web | Revisión de novedad (16-nov-2026) | Capítulo `e3_01_novedad` |

### Qué hicieron y verificaron los autores

- Definieron el problema, el protocolo preregistrado y sus enmiendas, y tomaron las decisiones metodológicas.
- Ejecutaron todos los entrenamientos y evaluaciones; las cifras del libro provienen de los archivos de resultados
  de esas corridas, no de estimaciones del asistente.
- Revisaron las conclusiones, incluido el resultado negativo (no se cumple R3), y las limitaciones declaradas.
- El asistente de IA no ejecutó los entrenamientos en GPU ni generó datos experimentales.

### Limitaciones de esta declaración

La búsqueda bibliográfica no incluyó Scopus ni IEEE Xplore. El asistente de IA puede cometer errores en código y
redacción; los autores revisaron el contenido, pero no se garantiza la ausencia total de fallos.

### Responsabilidad

Los autores asumen la responsabilidad del contenido de este trabajo, incluido el que se elaboró con apoyo de
herramientas de IA.

Fecha: 15 de noviembre de 2026

| Autor | Firma (nombre completo) |
|---|---|
| Mujica Camilo | Camilo Andres Mujica Escorcia |
| Romero Carlos | Carlos Isaac Romero Mendoza |
