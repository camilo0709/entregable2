# nnU-Net como pipeline propio del benchmark

**Fecha:** 2026-10-21 · **Código:** `codigo/entrenamiento_nnunet.py` ·
**Estado:** validado localmente (conversión + planificación + evaluación + las
3 ramas de creación de trainer); pendiente smoke de entrenamiento completo
(ver §6).

## 1. Por qué un pipeline aparte

nnU-Net (Isensee et al., *Nature Methods* 2021) no es "un modelo más" del
driver: es un **pipeline** que define su propia red, su parche, su batch, su
normalización y su protocolo de validación (5-fold CV). Meterlo en el bucle
del driver obligaría a forzar parámetros que contradicen su método. Por eso:

- sale del bucle de `driver_entrenamiento.py` (el driver lo excluye e imprime
  la instrucción de ejecutar este script);
- **reutiliza** del driver `guardar_resultado`, `recomponer_resumen` y
  `run_completado`, de modo que sus salidas son indistinguibles de un run
  normal y entran al resumen del benchmark sin excepción.

## 2. Equivalencia declarada: folds internos ↔ semillas del protocolo

El protocolo del Entregable 2 exige 10 modelos × 5 semillas
`[42, 123, 777, 2024, 31415]`. nnU-Net no usa semillas sueltas sino 5 folds;
la equivalencia, fijada en el código (`SEMILLAS` de `modelos_tb.py`), es:

| Fold interno nnU-Net | Semilla del protocolo | Carpeta de salida |
|---|---|---|
| 0 | 42 | `resultados/nnunet/seed_42/` |
| 1 | 123 | `resultados/nnunet/seed_123/` |
| 2 | 777 | `resultados/nnunet/seed_777/` |
| 3 | 2024 | `resultados/nnunet/seed_2024/` |
| 4 | 31415 | `resultados/nnunet/seed_31415/` |

Justificación: los 5 folds de nnU-Net particionan el pool de entrenamiento en
5 subconjuntos de validación disjuntos; cada fold produce un modelo entrenado
sin ver su subconjunto. Esto entrega exactamente 5 modelos independientes con
**varianza comparable** a la de 5 corridas con semillas distintas (en ambos
casos hay 5 entrenamientos sobre el mismo pool con particiones/azar
diferentes). La diferencia se declara en el informe: para los otros 9 modelos
la variación es de inicialización/aumento (misma partición train/val para
todos); para nnU-Net es de **partición** (además de inicialización). Es una
limitación de comparabilidad declarada, no oculta.

## 3. Presupuesto: qué es idéntico y qué difiere por diseño

| Elemento | Iguales con el resto del benchmark | Difiere (declarado) |
|---|---|---|
| Datos | mismo GT plata D1-R1 sellado; mismos splits por CSV (membresía §R1.8); mismas 194 train/val y 80 test_pos | conversión PNG/npz→NIfTI (reversible, solo lectura) |
| Presupuesto temporal | **trainer de 50 épocas** (`nnUNetTrainer_50epochs`), como todos | — |
| Resolución/canal | entrada 512×512, 1 canal (config `2d`) | parche `[512,512]` y batch `10` los fija el planificador de nnU-Net (verificado en los planes generados); los otros modelos usan su tamaño por defecto |
| Hardware | misma GPU de Colab, mismas sesiones | — |
| Métrica | Dice/IoU/etc. con `metricas_tb.py` idéntico | — |
| Validación reportada | `por_imagen.csv` = métricas de **val**, `por_imagen_test.csv` = test_pos, igual que el driver | la "val" de nnU-Net es su fold interno (pool de 194), no la partición val del CSV |

> **Declaración para el informe (§4/§6):** el presupuesto idéntico se garantiza
> en épocas, datos, hardware y métrica. El tamaño de parche/batch **no** se
> fuerza a igualdad porque forma parte del método de nnU-Net (su planificador);
> forzarlo convertiría a nnU-Net en "un U-Net con nombre raro". Esto se
> reporta como diferencia de diseño y se pondera en el análisis Pareto
> (§6: calidad vs coste).

## 4. GT, splits y convención de salidas

- El pool de 194 (train+val) se convierte a `imagesTr`/`labelsTr` con el
  split interno que define nnU-Net; los 80 `test_pos` (PNG) a `imagesTest`.
- Salida por fold (esquema idéntico al driver):

```
resultados/nnunet/seed_<semilla>/
├── resultado.json        # estado, dice_val, tiempos, meta (paper, familia…)
├── por_imagen.csv        # métricas por imagen de VAL (fold interno)
└── por_imagen_test.csv   # métricas por imagen de TEST (test_pos, GT plata)
```

- `recomponer_resumen` (driver) los recoge igual que cualquier otro modelo.

## 5. Uso (Colab)

```bash
# 0) una sola vez por sesión: la celda de setup de setup_colab.ipynb instala
#    nnunetv2 + setuptools (py3.12) y define NNUNET_RAW/PREPROC/RESULTS.

# 1) convertir + planificar (194+80 imágenes, ~20 s + plan ~1 min)
python -m codigo.entrenamiento_nnunet --solo-preparar

# 2) entrenar y evaluar los 5 folds (reanudable: omite folds con resultado.json)
python -m codigo.entrenamiento_nnunet

# 2b) un fold concreto / otro trainer
python -m codigo.entrenamiento_nnunet --folds 0 --trainer nnUNetTrainer_50epochs

# 3) resumen (lo hace el propio script al final; o manual):
python -c "import sys; sys.path.insert(0,'codigo'); \
from driver_entrenamiento import recomponer_resumen; recomponer_resumen()"
```

**Reanudación:** si la sesión de Colab cae, se re-ejecuta el mismo comando; los
folds con `resultado.json` se omiten (chequeo `run_completado`).

## 6. Verificado localmente / pendiente

| Paso | Estado |
|---|---|
| Conversión PNG/npz→NIfTI (194+80) + `dataset.json` | ✅ local |
| `plan_and_preprocess` (config 2d, `verify_dataset_integrity` OK) | ✅ local |
| `metricas_val_fold` (ruta `summary.json` y fallback) | ✅ local |
| `evaluar_test` (predicciones chequeadas por conteo) | ✅ local |
| `asegurar_trainer` (3 ramas: variante, custom, RuntimeError) | ✅ local |
| Smoke de entrenamiento 1 época × 2 iter (CPU, fold 0) | ⏸ interrumpido — riesgo bajo: `metricas_val_fold` tiene fallback si falta `summary.json` |

## 7. Amenazas a la validez declaradas (para el informe)

1. **Variación por partición vs semilla** (§2): la varianza de nnU-Net mide
   otra cosa que la de los otros modelos; se declara al comparar.
2. **Auto-planificación**: nnU-Net elige parche/batch; se pondera en el
   análisis Pareto (§6) y no se interpreta como "mejor arquitectura" sino
   como "mejor pipeline" (matriz §3.4 lo distingue: "pipeline ≠ red").
3. **Trainer acotado a 50 épocas**: no es el entrenamiento estándar de
   nnU-Net (300+1000 épocas); se declara para igualar presupuesto. Si
   nnU-Net rinde por debajo de lo publicado, esa es la causa esperada y está
   controlada.
