# Sección 8 · Reproducibilidad, ética y comunicación científica

## 8.1 Reproducibilidad del repositorio

### Repositorio e instalación

Repositorio versionado y público: **github.com/camilo0709/entregable2**
(historial con identidad de autor; cada push a `main` construye y publica el
informe vía GitHub Actions — `.github/workflows/libro.yml`).

| Entorno | Instalación exacta |
|---|---|
| **Informe** (local o CI) | `pip install -r libro/requirements.txt` → `jupyter-book build libro/` (~3 min; **ejecuta** el notebook EDA real por `execute_notebooks: force`) |
| **Benchmark** (Colab T4/L4 o local GPU) | `pip install -r requirements.txt` (versiones pinneadas) o `codigo/setup_colab.ipynb` en Colab |

**Versiones exactas** (`requirements.txt`, pinneadas): torch 2.2.2,
torchvision 0.17.2, numpy 1.26.4, scipy 1.13.1, pandas 3.0.6,
scikit-image 0.26.0, opencv-python-headless 4.10.0.84, pillow 13.0.0, timm
1.0.22, einops 0.8.2, nnunetv2 2.8.1, jupyter-book 1.0.4.post1, matplotlib
3.11.2, scikit-learn 1.9.1. Además — y esto es lo definitivo — **cada
`resultado.json` guarda las versiones realmente ejecutadas** en el campo
`entorno` (nuevo: `driver_entrenamiento.entorno_run()`), de modo que ninguna
métrica queda huérfana de su runtime.

### Semillas, configuraciones y particiones guardadas

- **Semillas:** `SEMILLAS = [42, 123, 777, 2024, 31415]` en
  `codigo/modelos_tb.py`; fijadas en torch, numpy y en el `DataLoader`
  (generator) del driver.
- **Configuración:** `CONFIG` del driver (150 épocas máx., early stopping 25,
  AdamW lr 1e-4 + cosine, batch 8, BCE+Dice 0,5/0,5, umbral 0,5) guardada en
  cada `resultado.json` (`config`); nnU-Net usa su propia configuración
  (5 folds ↔ semillas, `docs/NNUNET_PIPELINE.md`).
- **Particiones:** los 5 CSV sellados con SHA-256 en
  `datos/splits/split_manifest_v2.json`; el EDA los verifica al arrancar y el
  check C6 de la replicación confirma que son subconjunto del trainval
  oficial.

### Scripts que regeneran todas las tablas y figuras

| Artefacto | Script regenerador |
|---|---|
| EDA (figuras A1–A9 de la Sección 2) | `libro/book/EDA_Decisiones_TBX11K.ipynb` (se reejecuta en cada build) |
| PRISMA + esquemas de la Sección 3 | `codigo/build_figuras_rev.py` |
| Replicación del paper guía (Tablas de la Sección 5) | `codigo/replicacion_paper_guia.py` |
| Tablas A/B/C, ranking, Pareto, inferencia estadística (Secciones 6–7) | `codigo/estadistica_tb.py` sobre `resultados/` (corrida final pendiente al cerrar Colab; la estructura de entrada está fijada por el driver) |

### Registro de experimentos

Cada corrida deja en `resultados/<modelo>/seed_<s>/`:

```
resultado.json    # estado, dice_val, época óptima, segundos, config, entorno
historial         # curva loss/dice por época (dentro de resultado.json)
por_imagen.csv    # métricas por imagen de val (entrada de la estadística)
mejor_modelo.pt   # checkpoint de la época óptima
```

más `resumen_runs.csv` agregado (regenerable desde los JSON individuales con
`recomponer_resumen()`: el resumen se deriva de los archivos de run, nunca al
revés). El driver es **reanumable** (una corrida "hecha" = JSON válido), así
que el registro sobrevive a desconexiones de sesión.

## 8.2 Lista de verificación de reproducibilidad

Anexa al informe: **[Anexo · Lista de verificación de reproducibilidad](anexo_checklist_reproducibilidad.md)**
(33 ítems sobre datos, entrenamiento, código y ética, cada uno con estado y
evidencia apuntando al artefacto exacto). Resumen: 32 cumplidos · 1 con
estructura entregada y valores pendientes de la fase 6 · 0 incumplimientos.

## 8.3 Ética y uso responsable

### Sesgo y equidad

- **Sesgo de adquisición (medido):** el EDA A4 separa sanos de TB con AUC
  0,842 usando solo estadísticos de apariencia de la imagen (p-permutación
  0,010): parte del "éxito" de cualquier modelo puede venir del **escáner**,
  no de la patología. Mitigación: `test_neg` (40 sanos) se usa solo como
  control de falsos positivos, nunca dentro de la pérdida ni del score
  principal; se reporta explícitamente.
- **Equidad por subgrupo: no auditable aquí.** El paquete liberado de TBX11K
  no expone sexo/edad (el paper sí los tuvo). Se declara la imposibilidad de
  evaluar equidad por subgrupo en lugar de fingir que no aplica; cualquier
  despliegue real exigiría esos metadatos.
- **Sesgo del GT:** las máscaras de plata las produce SAM 2 con el caja del
  experto como prompt; su sesgo sistemático (bordes suaves) está acotado con
  el techo humano medido (0,9879) — ninguna afirmación de "mejor que el
  humano" es interpretable por encima de ese techo.
- **Prevalencia:** train/val es 100 % TB positivo (por diseño del subset);
  las tasas de detección clínicas NO son las de nuestro test.

### Privacidad

Radiografías públicas y **de-identificadas** por sus proveedores (los
conjuntos subyacentes, Jaeger et al. 2014 y Chauhan et al. 2014, se
publicaron con finalidad de investigación y exentos por sus instituciones;
el paper declara el mismo procedimiento para TBX11K). El paquete no contiene
nombres ni identificadores de paciente (de hecho, tampoco ID de paciente).
Este repositorio **no redistribuye imágenes ni máscaras derivadas**: solo
splits (IDs), hashes, código y figuras agregadas (`.gitignore`).

### Consecuencias de errores del modelo

| Error | Consecuencia clínica | Respuesta del diseño |
|---|---|---|
| **Falso negativo** (no segmentar una lesión real) | Retraso de tratamiento, mayor transmisión | Recall reportado por estrato (lesiones pequeñas A7); el modelo es **asistente de tamizaje**, nunca valla diagnóstica |
| **Falso positivo** (segmentar sano/no-TB) | Estrés, estudios de seguimiento innecesarios | Control OOD dedicado (`test_neg`); tasa de FP reportada aparte, nunca promediada con positivos |

**Uso previsto:** priorización de estudio bajo revisión humana en escenarios
de baja dotación de radiólogos (política OMS de CXR+CAD, 2021).
**Fuera de alcance declarado:** diagnóstico autónomo, uso en poblaciones o
modalidades no representadas en TBX11K, cuantificación de progreso.

### Cumplimiento de licencias

| Activo | Licencia | Cumplimiento |
|---|---|---|
| TBX11K (datos) | **Sin licencia formal**; exige citación (`liu2020rethinking`) | Citado en Secciones 1 y 5; imágenes no redistribuidas; ausencia de licencia declarada en vez de asumir permisos |
| Montgomery/Shenzhen; DA/DB | Publicados con fines de investigación (Jaeger 2014; Chauhan 2014) | Citados; no redistribuidos |
| SAM 2 (solo generación del GT) | Apache-2.0 | Solo checkpoint de uso; no es modelo evaluado; SHA-256 registrado |
| Pesos preentrenados | — | **El benchmark no usa ninguno** (`encoder_weights: None`): no hay redistribución de pesos de terceros |
| PyTorch 2.2.2 | BSD-3-Clause | OK |
| MONAI 1.x | Apache-2.0 | OK (UNETR/SegResNet) |
| segmentation-models-pytorch | MIT | OK |
| nnU-Net v2 (2.8.1) | La metadata del paquete instalado declara Apache-2.0 | OK |
| timm | Apache-2.0 | OK |

## 8.4 Comunicación científica

- **Registro:** redacción académica formal en todo el informe; la terminología
  (Dice, APbb, plata/bronce, cota humana) se define en la Sección 1 y se usa
  de forma consistente de la 1 a la 8.
- **Figuras y tablas autocontenidas:** cada figura del EDA (A1–A9), el PRISMA
  y los esquemas de la Sección 3 llevan título, unidades (px, %, Dice),
  leyenda y **fuente** (script que la genera); las tablas de la Sección 5
  declaran "paper" vs "medido" con Δ explícito.
- **Bibliografía:** `referencias/references.bib` — BibTeX con **DOI verificado
  contra `doi.org`** (2026-10-21) o eprint de arXiv verificado contra su API;
  política explícita: *sin referencias no verificables* (cada entrada de las
  Secciones 3 y 5 puede rastrearse hasta su identificador resuelto). El
  fichero `referencias/REFERENCIAS_BIBLIOGRAFICAS.md` conserva el detalle de
  la verificación.

## 8.5 Entregables y defensa oral

### Mapa rúbrica → evidencia

| Entregable de la rúbrica | Dónde está |
|---|---|
| Informe en Jupyter Book con ejecución reproducible | `libro/` → https://camilo0709.github.io/entregable2/ (CI ejecuta celdas en cada build) |
| Código completo y documentado (EDA, 10 modelos, baselines, ajuste/sobreajuste, estadística, robustez, interpretabilidad) | `codigo/` (Secciones 2, 4, 6, 7 del informe) |
| Dataset o instrucciones verificables + Data Card | `README.md` Sección 3 + Sección 1.4 (Data Card con hashes) |
| Ficha de ruta metodológica (modalidad, recorrido I1–I8, evidencia de Tablas A/B/C) | Sección 7 (pestaña de resultados al cerrar Colab) |
| Tabla comparativa (métricas, costos, inferencia estadística) | Sección 6 (ídem; datos de entrada: `resultados/`) |
| Registro de experimentos (configs, semillas, resultados por ejecución) | `resultados/<modelo>/seed_<s>/` (estructura en 8.1) |
| Presentación oral con defensa técnica | Documento de preparación oral (guion + preguntas anticipadas), entregado aparte; las figuras del libro sirven como diapositivas |
