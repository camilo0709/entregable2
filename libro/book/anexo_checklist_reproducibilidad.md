# Anexo · Lista de verificación de reproducibilidad

> Rellena y firmada por el equipo, según la rúbrica del Entregable 2
> ("Lista de verificación de reproducibilidad completada y anexa").
> Leyenda: ✅ cumplido · ⏳ pendiente de resultado final (estructura entregada) ·
> N/A no aplica (con justificación).

## A. Datos y partición

| # | Ítem | Estado | Evidencia |
|---|---|---|---|
| A1 | El dataset es de acceso público o se dan instrucciones verificables para obtenerlo | ✅ | `README.md` (sección "Datos"); TBX11K oficial (Liu et al., CVPR 2020) |
| A2 | Las imágenes no se redistribuyen y se documenta su licencia | ✅ | `.gitignore`; README "Licencias"; Sección 8.3 |
| A3 | Partición train/val/test definida antes de entrenar y **sellada** (hashes) | ✅ | `datos/splits/split_manifest_v2.json` (5 × SHA-256, 2026-10-20) |
| A4 | Verificación automática de integridad de la partición en cada ejecución | ✅ | EDA A8 (falla si un CSV cambia); réplica C6 (0 fugas) |
| A5 | Sin *data leakage* entre particiones (duplicados auditados) | ✅ | EDA A2: 0 MD5, 29 pHash auditados uno a uno |
| A6 | Protocolo de etiquetado y ruido de etiqueta declarados y cuantificados | ✅ | Sección 1.4 (plata SAM 2, techo 0,9879) |

## B. Entrenamiento y evaluación

| # | Ítem | Estado | Evidencia |
|---|---|---|---|
| B1 | ≥ 5 ejecuciones independientes por modelo | ✅ | `SEMILLAS = [42, 123, 777, 2024, 31415]` en `modelos_tb.py` |
| B2 | Presupuesto de optimización idéntico entre modelos | ✅ | `CONFIG` único del driver (150 ép., ES 25, AdamW 1e-4 + cosine, BCE+Dice 0,5/0,5, umbral 0,5) — Sección 4 |
| B3 | Sin pesos preentrenados (igualdad de condiciones) | ✅ | `encoder_weights: None` en los 10; verificado en `modelos_tb.py` |
| B4 | Semillas fijadas y aplicadas (torch, numpy, DataLoader generator) | ✅ | `driver_entrenamiento.py` (`semilla_fn`, `g.manual_seed`) |
| B5 | Configuración completa registrada por run | ✅ | `resultado.json` (campo `config`) |
| B6 | Versiones exactas del runtime registradas por run | ✅ | `resultado.json` (campo `entorno`, escrito por `entorno_run()`) |
| B7 | Curvas de pérdida/métrica por época guardadas | ✅ | `resultado.json` (campo `historial`) |
| B8 | Métricas por imagen disponibles (no solo agregados) | ✅ | `por_imagen.csv` por run |
| B9 | Pruebas de significancia con corrección por comparaciones múltiples | ⏳ | `codigo/estadistica_tb.py` (Holm; estructura en Sección 1.2, resultados en Sección 6 al cerrar los runs) |
| B10 | Intervalos de confianza (bootstrap) y tamaño de efecto | ⏳ | Ídem (Sección 6) |
| B11 | Análisis de sobreajuste (curvas train/val, early stopping) | ✅ | `historial` + early stopping declarado; análisis en Sección 7 |
| B12 | Pruebas de calibración y robustez de la ruta | ⏳ | Sección 7 (tras resultados) |

## C. Código y entorno

| # | Ítem | Estado | Evidencia |
|---|---|---|---|
| C1 | Repositorio versionado con historial público | ✅ | `github.com/camilo0709/entregable2` |
| C2 | Instrucciones de instalación paso a paso | ✅ | `README.md` (Secciones 1–2) |
| C3 | Dependencias con versiones exactas | ✅ | `requirements.txt` (pinneado) + registro por run (B6) |
| C4 | Entorno reproducible preferente (contenedor/CI) | ✅ | CI de GitHub Actions con entorno limpio en cada build del libro (`.github/workflows/libro.yml`); Colab `setup_colab.ipynb` |
| C5 | Scripts que regeneran figuras y tablas | ✅ | Tabla en `README.md` §4 (EDA, `build_figuras_rev.py`, `replicacion_paper_guia.py`, `estadistica_tb.py`) |
| C6 | Código del EDA, modelos, baselines y estadística completo | ✅ | `codigo/` (10 modelos + 3 baselines en `modelos_tb.py`) |
| C7 | El informe se construye ejecutando las celdas (no solo markdown) | ✅ | `execute_notebooks: force` en `libro/_config.yml`; CI ejecuta el EDA real |
| C8 | Registro de experimentos (configs, semillas, resultados por ejecución) | ⏳ | Estructura `resultados/<modelo>/seed_<s>/` + `resumen_runs.csv`; se llena al descargar los runs de Colab |

## D. Ética y comunicación

| # | Ítem | Estado | Evidencia |
|---|---|---|---|
| D1 | Riesgos de sesgo y equidad discutidos | ✅ | Sección 8.3 (sesgo de adquisición AUC 0,842; demografía no expuesta) |
| D2 | Privacidad de los datos tratada | ✅ | Sección 8.3 (de-identificados, sin ID paciente, no redistribución) |
| D3 | Consecuencias de errores del modelo discutidas | ✅ | Sección 8.3 (FN/FP clínicos, uso asistencial, no autónomo) |
| D4 | Licencias de datos y de librerías/pesos cumplidas y documentadas | ✅ | Sección 8.3 + `README.md` |
| D5 | Figuras y tablas autocontenidas (unidades, leyenda, fuente) | ✅ | Política en Sección 8.4; aplicada en EDA (A1–A9) y Sección 3 |
| D6 | Bibliografía verificable (DOI/arXiv comprobados), sin referencias inventadas | ✅ | `referencias/references.bib` (DOIs verificados contra doi.org, 2026-10-21) |
| D7 | Limitaciones y declaraciones post-hoc explícitas | ✅ | Secciones 1 (prereg), 3 (gap), 5 (elección post-hoc) |

**Resumen:** 29 ✅ · 4 ⏳ (dependen de terminar los runs de Colab; la
estructura y los scripts están entregados) · 0 incumplimientos.
