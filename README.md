# TBX11K · Benchmark de Deep Learning para segmentación de tuberculosis

Entregable 2 del curso de Deep Learning: **10 modelos de segmentación × 5
semillas** sobre el dataset TBX11K, con inferencia estadística, revisión
sistemática, replicación del paper guía y GT de "plata" validado.

📖 **Informe (Jupyter Book) publicado:**
**https://camilo0709.github.io/entregable2/**
(build automático con cada push a `main` — ver `.github/workflows/libro.yml`)

---

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `libro/` | Proyecto Jupyter Book (fuente del informe). `book/` = secciones 1–8; `_config.yml` ejecuta los notebooks en cada build |
| `codigo/` | EDA (`eda_lib_tb.py`), registro de modelos (`modelos_tb.py`), driver de entrenamiento, métricas, datasets, nnU-Net, estadística, figuras |
| `datos/splits/` | **Sellados**: 5 CSV + `split_manifest_v2.json` con SHA-256. Jamás se modifican (solo lectura verificada en cada EDA) |
| `datos/ground_truth_test/` | GT de plata del test (80 PNG) + `ANOTACION_METRICAS.json` (techo humano 0,9879) |
| `docs/` | Decisiones (D1-R1), protocolo de anotación, pipeline nnU-Net |
| `referencias/` | `REFERENCIAS_BIBLIOGRAFICAS.md`, `references.bib` (BibTeX), réplica del paper guía |
| `.github/workflows/` | CI: construye el libro (ejecuta celdas) y publica en Pages |

Los **datos pesados no se redistribuyen** (imágenes TBX11K ≈ 4 GB, pesos ≈
355 MB): `.gitignore` los excluye; abajo están las instrucciones para
obtenerlos.

---

## 1. Construir el informe (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r libro/requirements.txt
jupyter-book build libro/        # ~3 min: ejecuta el notebook EDA real
open libro/_build/html/index.html
```

En CI no hace falta: cada `git push` a `main` que toque `libro/`, el EDA o
los modelos reconstruye y republica el sitio automáticamente.

## 2. Reproducir el benchmark (Colab, T4/L4)

1. Sube el proyecto a Google Drive y abre `codigo/setup_colab.ipynb` en
   Colab: instala dependencias (`requirements.txt`) y valida el entorno.
2. Ejecuta la celda de entrenamiento del driver. **Reanudable**: si la sesión
   cae, re-ejecútala; las corridas terminadas se saltan (`resultado.json`
   válido = hecha). 50 corridas (10 modelos × 5 semillas) + baselines.
3. Al terminar (0 corridas pendientes), descarga `resultados/`:
   - `resultados/<modelo>/seed_<s>/resultado.json` — métricas + config +
     **entorno** (versiones exactas del runtime),
   - `por_imagen.csv` — métricas por imagen (para inferencia estadística),
   - `resumen_runs.csv` — tabla agregada (regenerable desde los JSON).

Configuración idéntica para todos los modelos (`driver_entrenamiento.py`):
150 épocas máx., early stopping 25, AdamW 1e-4 + cosine, batch 8 (10 en
nnU-Net), pérdida BCE+Dice 0,5/0,5, umbral 0,5, semillas `[42, 123, 777,
2024, 31415]`, sin pesos preentrenados (`encoder_weights=None`, igualdad de
presupuesto).

## 3. Datos (obtención verificable)

1. **TBX11K** — dataset original de Liu et al., CVPR 2020 (paquete oficial
   con `README.md` citable): 11 200 radiografías 512×512 con cajas de TB.
   Colócalo en `datos/kaggle_dataset/TBX11K/` conservando la estructura
   `imgs/`, `annotations/`, `lists/`.
2. **Splits sellados** — los 5 CSV de `datos/splits/` están versionados en
   este repo; su integridad se verifica con los SHA-256 de
   `split_manifest_v2.json` (el EDA falla si un byte cambia).
3. **GT de plata** — las 274 pseudomáscaras SAM 2 (165 train + 29 val + 80
   test) se regeneran con `codigo/generar_pseudo_gt.py`; el manifiesto
   `pseudo_gt_manifest_v2.json` guarda SHA-256 del script y del checkpoint.
   La validación del GT de test está en
   `datos/ground_truth_test/ANOTACION_METRICAS.json`.
4. La **Data Card** completa (tamaño, estructura, protocolo de etiquetado,
   ruido, hashes) está en la **Sección 1** del informe.

## 4. Regenerar figuras y tablas

| Artefacto | Script |
|---|---|
| EDA completo (figuras A1–A9) | `libro/book/EDA_Decisiones_TBX11K.ipynb` (se ejecuta en cada build) |
| PRISMA + esquemas de la revisión | `codigo/build_figuras_rev.py` |
| Replicación del paper guía | `codigo/replicacion_paper_guia.py` |
| Tablas A/B/C e inferencia estadística | `codigo/estadistica_tb.py` (corre sobre `resultados/`) |

## Licencias y uso responsable

- **Datos:** TBX11K no declara licencia formal en su paquete; exige citación
  (`liu2020rethinking`). Las imágenes **no** se redistribuyen en este repo.
  Los conjuntos subyacentes (Montgomery/Shenzhen, Jaeger et al. 2014; DA/DB,
  Chauhan et al. 2014) son públicos con fines de investigación.
- **Pesos preentrenados:** este benchmark **no usa ninguno** (presupuesto
  igualado). SAM 2 (Apache-2.0) se usó solo para generar el GT de plata, no
  es un modelo evaluado.
- **Librerías:** PyTorch (BSD-3), MONAI (Apache-2.0),
  segmentation-models-pytorch (MIT), nnU-Net v2 (la metadata del paquete
  2.8.1 declara Apache-2.0), timm (Apache-2.0).
- **Uso:** asistencia al radiólogo en tamizaje — **no** diagnóstico
  autónomo. Ver Sección 8 (ética) del informe.

## Citar este trabajo

```bibtex
@misc{entregable2tbx11k,
  title  = {Benchmark de Deep Learning para segmentación de tuberculosis sobre TBX11K},
  author = {{Entregable 2 — Curso de Deep Learning}},
  year   = {2026},
  url    = {https://camilo0709.github.io/entregable2/},
  note   = {Código: https://github.com/camilo0709/entregable2}
}
```
