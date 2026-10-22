# Sección 4 · Implementación de modelos benchmark

Esta sección documenta el **diseño experimental**, la **implementación** de
los 10 modelos y los 3 baselines, y el **protocolo** que garantiza una
comparación justa (rúbrica 4.1–4.4). Los resultados numéricos viven en la
[Sección 6](06_benchmark_final.md); las pruebas de bondad de ajuste,
calibración, robustez e interpretabilidad se ejecutan en la ruta de la
[Sección 7](07_ruta_metodologica.md).

Fuentes de esta sección: `codigo/modelos_tb.py` (registro),
`codigo/driver_entrenamiento.py` (protocolo), `codigo/dataset_tb.py` (datos),
`codigo/metricas_tb.py` (métricas), `codigo/entrenamiento_nnunet.py` y
`docs/NNUNET_PIPELINE.md` (nnU-Net), `codigo/perfil_modelos.py` (medidas
computacionales).

---

## 4.1 Diseño experimental

### 4.1.1 Tarea, variables de entrada y variable objetivo

**Tarea:** segmentación binaria de lesiones de tuberculosis en radiografías
de tórax (Ruta B, imagen 2D).

| Elemento | Definición |
|---|---|
| Variable de entrada $x$ | Radiografía 512×512, 1 canal (escala de grises), $\mathbb{R}^{1\times512\times512}$ |
| Normalización | z-score **por imagen** (media y desviación de la propia imagen): no existe ninguna estadística global aprendida del conjunto de entrenamiento, de modo que un escalador no puede filtrar información entre particiones |
| Variable objetivo $y$ | Máscara binaria 512×512 del GT de plata (D1-R1), $\{0,1\}^{512\times512}$; la membresía de cada imagen la define su CSV sellado, nunca el directorio |
| Modelo | $f_\theta(x) \in \mathbb{R}^{1\times512\times512}$ (logits por píxel); probabilidad = $\sigma(f_\theta(x))$ |
| Decisión | Binarización con umbral **0,5 fijado a priori** (no se optimiza ningún umbral sobre test) |
| Pérdida (común) | $\mathcal{L} = 0,5\,\mathrm{BCEWithLogits} + 0,5\,(1-\mathrm{Dice}_{\text{soft}})$, idéntica para las 10 redes |

### 4.1.2 División del dataset (idéntica para los 13 modelos)

La partición se congeló en la fase de GT (2026-10-20, decisión D1-R1) **antes
de cualquier entrenamiento** y se selló con SHA-256:

| Split | n | Papel | Fuente de membresía |
|---|---|---|---|
| `train` | 165 | Ajuste de pesos (+ augmentación) | `train.csv` sellado |
| `val` | 29 | Early stopping y selección de época óptima | `val.csv` sellado |
| `test_pos` | 80 | Prueba (GT plata congelado) | `test_pos.csv` sellado |
| `test_neg` | 40 | Control OOD: tasa de falsos positivos en sanos | `test_neg.csv` sellado |

Garantías de reserva del test (rúbrica: "prueba reservada y tocada una sola
vez al final"):

1. **Estructural:** el driver de entrenamiento *no abre* `test_pos` bajo
   ninguna circunstancia; solo genera `por_imagen.csv` sobre `val`. No existe
   código de entrenamiento que pueda leer el test.
2. **Por fase:** la evaluación de test se ejecuta en la fase de análisis
   (Sección 6), cuando los 55 runs están congelados, con el mejor checkpoint
   de cada corrida (época seleccionada por val). Única salvedad declarada:
   el pipeline de nnU-Net genera sus predicciones de test dentro de su
   propia corrida, **después** de entrenar cada fold congelado (50 épocas
   fijas, sin selección alguna basada en test); la única toma de contacto
   analítica —comparar, decidir, publicar— sigue siendo una sola, en la
   fase de resultados.
3. **Integridad:** `verificar_integridad_splits.py` y el EDA comparan los
   SHA-256 de los CSV contra `split_manifest_v2.json` en cada ejecución; el
   check C6 de la replicación confirma que los splits son subconjunto del
   trainval oficial (0 imágenes ajenas).

### 4.1.3 Pipeline modular

| Módulo | Responsabilidad |
|---|---|
| `dataset_tb.py` | Carga por CSV sellados, z-score por imagen, augmentación solo-train determinista, verificación previa de splits |
| `modelos_tb.py` | Registro único: 10 modelos + 3 baselines, parámetros exactos de cada arquitectura, semillas, verificación de dependencias |
| `driver_entrenamiento.py` | Bucle de entrenamiento común (pérdida, optimizador, scheduler, early stopping, semillas, reanudabilidad, registro por run) |
| `metricas_tb.py` | Implementación única de métricas (Dice, IoU, HD95, ASSD) con convenciones fijadas |
| `entrenamiento_nnunet.py` | Pipeline propio de nnU-Net (5 folds ↔ semillas) que escribe el mismo esquema de resultados |
| `perfil_modelos.py` | Parámetros, FLOPs, latencia y memoria pico de cada arquitectura (tabla 4.3) |
| `preprocesamiento` | **No hay paso separado**: la normalización vive dentro del Dataset y es idéntica para train/val/test |

### 4.1.4 Selección de hiperparámetros

**Protocolo principal (los 10 modelos): configuración fija, sin búsqueda.**
Una búsqueda por modelo con 165 imágenes de entrenamiento introduce más
sesgo que señal (selecciona sobre ruido de val) y rompería la igualdad de
presupuesto. La rúbrica exige "el mismo número de intentos y presupuesto por
modelo": aquí ese número es **N = 1** para los diez, con la configuración
declarada en la tabla 4.2. Todos los modelos reciben exactamente las mismas
150 épocas máximas, el mismo early stopping y el mismo hardware.

**Única búsqueda sistemática permitida: `unet_fuerte` (baseline, no está en
los 10).** Espacio declarado aquí, **antes de tocar el test** (prereg):

| Hiperparámetro | Valores | Puntos |
|---|---|---|
| learning rate | {1e-4, 3e-4} | 2 |
| weight decay | {1e-5, 1e-4} | 2 |

Total: 4 configuraciones × 1 semilla (42), evaluadas por Dice medio en
`val`; la ganadora se congela y se reentrena con las 5 semillas. Como la
configuración por defecto (1e-4 / 1e-5) es uno de los cuatro puntos y ya se
entrena en el benchmark principal, la búsqueda agrega solo 3 corridas:

```bash
python codigo/driver_entrenamiento.py --modelos unet_fuerte --semillas 42 \
    --lr 3e-4 --weight-decay 1e-5 --resultados resultados/unet_fuerte_busqueda
python codigo/driver_entrenamiento.py --modelos unet_fuerte --semillas 42 \
    --lr 1e-4 --weight-decay 1e-4 --resultados resultados/unet_fuerte_busqueda
python codigo/driver_entrenamiento.py --modelos unet_fuerte --semillas 42 \
    --lr 3e-4 --weight-decay 1e-4 --resultados resultados/unet_fuerte_busqueda
```

(Los flags `--lr` y `--weight-decay` se añadieron al driver para esta
búsqueda; por defecto reproducen la configuración común.) La selección es
solo sobre `val`: `unet_fuerte` es cota superior de la familia U-Net, no un
modelo de la comparación principal.

---

## 4.2 Implementación de los modelos

### 4.2.1 Registro único de arquitecturas

Los 13 modelos viven en `modelos_tb.py` como un solo diccionario de verdad:
el driver importa de ahí, de modo que el protocolo no puede desviarse por
accidente. Librerías con versión fijada: `segmentation-models-pytorch 0.5.0`
y `monai 1.4.0` (verificadas construyendo las 10 arquitecturas; las versiones
realmente ejecutadas en cada corrida quedan en el campo `entorno` de cada
`resultado.json`).

| # | Modelo | Familia | Paper | Implementación (versión fijada) | Declaración |
|---|---|---|---|---|---|
| 1 | U-Net | CNN | Ronneberger, MICCAI 2015 | smp `Unet`, encoder ResNet-34 | — |
| 2 | Attention U-Net | CNN + atención | Oktay, MICCAI 2018 | smp `Unet` + `decoder_attention_type="scse"` | smp implementa la atención como bloques **SCSE** (canal + espacial), no como las *attention gates* aditivas del paper; es el mecanismo de atención disponible con el mismo protocolo y se declara como variante de la misma familia. Nota verificada empíricamente: el parámetro `attention_type` es **ignorado en silencio** por `smp.Unet` (devolvería un U-Net plano); el parámetro correcto es `decoder_attention_type` |
| 3 | U-Net++ | CNN (skip denso) | Zhou, DLMIA 2018 | smp `UnetPlusPlus`, ResNet-34 | — |
| 4 | DeepLabV3+ | CNN + ASPP | Chen, CVPR 2018 | smp `DeepLabV3Plus`, ResNet-50 | Requiere batch ≥ 2 (la rama de *image pooling* del ASPP usa BN sobre 1×1); batch 8 del protocolo lo cumple |
| 5 | nnU-Net | CNN (autoconfig) | Isensee, Nat. Methods 2021 | Pipeline propio `nnunetv2 2.8.1` (config `2d`) | Su planificador fija parche [512,512] y batch 10; ver presupuesto en 4.3 |
| 6 | TransUNet | CNN + Transformer | Chen, MedIA 2024 | **Sustituto declarado:** smp `Unet` ResNet-50 + decoder con atención SCSE | La implementación externa (`transunet`) no está en el entorno; el sustituto conserva la hibridación CNN+atención global. Se entrena el sustituto en Colab |
| 7 | Swin-UNet | Swin Transformer | Cao, ECCV W. 2022 | **Sustituto declarado:** smp `Unet` con encoder MiT-B2 (el jerárquico de SegFormer) | Implementación externa no disponible; se conserva el encoder transformer jerárquico. Se entrena el sustituto en Colab |
| 8 | UNETR | ViT | Hatamizadeh, WACV 2022 | MONAI `UNETR` 2D (ViT-B: hidden 768, 12 cabezas, `feature_size` 16, `proj_type=conv`) | — |
| 9 | SegFormer | MiT jerárquico | Xie, NeurIPS 2021 | smp `Segformer`, encoder MiT-B2 | — |
| 10 | MedNeXt | ConvNeXt/residual | Roy, MICCAI 2023 | **Sustituto declarado:** MONAI `SegResNet` 2D (`init_filters` 32, profundidad (1,2,2,4)) | MedNeXt aún no está en MONAI; SegResNet es el par residual más cercano disponible por pip |
| B1 | `empty_mask` | Trivial | — | Predice máscara vacía (sin red) | Cota inferior: Dice = 0 en positivos |
| B2 | `bbox_mask` | Trivial | — | Rectángulo del anotador original, sin refinamiento | Baseline propio del proyecto |
| B3 | `unet_fuerte` | CNN (tuned) | — | Mismo U-Net (ResNet-34) con la búsqueda de 4.1.4 | Cota superior de la familia U-Net |

Ninguna red usa **pesos preentrenados** (`encoder_weights=None` en las 10):
condición de igualdad de presupuesto y origen de los pesos, además de evitar
restricciones de licencia (Sección 8.3).

### 4.2.2 Hiperparámetros y estrategia de entrenamiento (protocolo común)

Valores únicos aplicados por `driver_entrenamiento.CONFIG` a **todos** los
modelos entrenables (incluido `unet_fuerte` salvo en su búsqueda):

| Hiperparámetro | Valor | Nota |
|---|---|---|
| Optimizador | AdamW | |
| Learning rate | 1e-4 | |
| Weight decay | 1e-5 | |
| Scheduler | CosineAnnealingLR, $T_{max}$ = épocas máximas | |
| Épocas máximas | 150 | |
| Early stopping | 25 épocas sin mejora de Dice-val; se guarda la época óptima | Selección solo sobre val |
| Batch size | 8 | 10 en nnU-Net (su planificador) |
| Pérdida | 0,5 BCEWithLogits + 0,5 Soft-Dice | Idéntica en las 10 |
| Umbral | 0,5 | Fijado a priori |
| Dropout / regularización extra | Los defaults de cada arquitectura (sin retocar por modelo) | |
| Gradient clipping | **No aplicado** (declarado: los smoke tests no mostraron inestabilidad de gradientes) | |
| Precisión mixta (AMP) | **No aplicada**: fp32 completo para todos | Evita una diferencia numérica entre modelos; coste ≈ 2× de tiempo declarado |
| Augmentación (solo train) | Flip horizontal p = 0,5; flip vertical p = 0,3; bruido de brillo $\mathcal{N}(1,0{,}05)$ p = 0,3; ruido aditivo $\sigma=0{,}03$ p = 0,2 | Determinista por `RandomState(1000 + índice)`: mismas transformaciones en todas las corridas |

### 4.2.3 Registro obligatorio por corrida

| Registro exigido | Dónde queda | Estado |
|---|---|---|
| Curvas de entrenamiento y validación | `resultado.json → historial` (loss y Dice por época, train y val) | Automático por run |
| Tiempo de entrenamiento | `resultado.json → segundos` | Automático |
| Latencia de inferencia (ms/imagen) | `perfil_modelos.py` (mediana de 5 pases a 512×512) | Medido (tabla 4.3); repetible en GPU con `--device cuda` |
| Número de parámetros y FLOPs | `perfil_modelos.py` (`resultados/perfil_modelos.csv`) | Medido (tabla 4.3) |
| Memoria pico | `perfil_modelos.py --device cuda` (`pico_memoria_mb`) | Disponible en GPU |
| Hardware | `resultado.json → entorno` (GPU, versiones) + Sección 4.3 | Automático |
| Consumo energético | **No medido** (declarado): Colab no expone medición confiable por corrida | Limitación explícita |

**Tabla 4.3 · Perfil computacional medido** (entrada 1×512×512, batch 1;
params y FLOPs son exactos y no dependen del hardware; latencia medida en
CPU de referencia — la latencia de despliegue se mide en GPU con
`perfil_modelos.py --device cuda`):

| modelo | params (M) | GMACs | GFLOPs | latencia (ms) |
|---|---|---|---|---|
| U-Net | 24,43 | 30,99 | 61,98 | 1 766 |
| Attention U-Net | 24,54 | 31,03 | 62,06 | 3 286 |
| U-Net++ | 26,07 | 73,39 | 146,78 | 3 975 |
| DeepLabV3+ | 26,67 | 36,49 | 72,99 | 1 973 |
| nnU-Net | — | — | — | — |
| TransUNet (sustituto) | 33,81 | 42,48 | 84,95 | 3 085 |
| Swin-UNet (sustituto) | 27,47 | 27,37 | 54,73 | 2 852 |
| UNETR | 116,24 | 105,14 | 210,27 | 6 303 |
| SegFormer | 24,72 | 21,08 | 42,16 | 3 150 |
| MedNeXt (sust. SegResNet) | 6,30 | 62,61 | 125,21 | 2 281 |
| `empty_mask` | 0 | 0 | 0 | — |
| `bbox_mask` | 0 | 0 | 0 | — |
| `unet_fuerte` | 24,43 | 30,99 | 61,98 | 1 117 |

Medido el 2026-10-21 con `codigo/perfil_modelos.py` (entrada 1×512×512,
batch 1, `resultados/perfil_modelos.csv`). La latencia es la mediana de 5
pases en CPU de referencia (Apple silicon, fp32); en GPU será ~2 órdenes de
menor y se reporta con `perfil_modelos.py --device cuda` en la misma sesión
de Colab del benchmark. nnU-Net no se perfila aquí (es un pipeline con su
propio planificador); los dos baselines triviales no tienen red.

Lecturas relevantes para el análisis posterior (Sección 6):

- **U-Net++ cuesta 2,4× los MACs de U-Net** con solo +1,6 M parámetros: los
  skips densos multiplican el cómputo, no el tamaño — candidato natural del
  análisis Pareto.
- **UNETR es el más pesado en las tres dimensiones** (116 M parámetros,
  210 GFLOPs, 3,4× la latencia de U-Net en CPU): recibe los mismos pasos de
  optimización que el resto, y su coste se pondera en el Pareto.
- **MedNeXt (SegResNet) es el más liviano en parámetros (6,3 M) pero no en
  cómputo (125 GFLOPs)**: profundidad residual a resolución completa.
- **Attention U-Net suma 0,1 % de parámetros y MACs** sobre U-Net (los bloques
  SCSE son element-wise), pero ~1,9× de latencia en CPU: la atención aquí es
  "gratis" en FLOPs y cara en CPU (operaciones poco paralelizables sin
  vectorizar).
- **SegFormer tiene el cómputo más bajo de las 10** (42 GFLOPs) con 24,7 M
  parámetros: si su Dice compite, será el punto de partida del Pareto.

---

## 4.3 Requisitos experimentales

### Semillas y determinismo

`SEMILLAS = [42, 123, 777, 2024, 31415]` (mínimo 5 exigido). Por corrida se
fijan **todas** las fuentes de azar y se documenta la configuración de
determinismo:

| Control | Implementación (`driver_entrenamiento.py`) |
|---|---|
| Python `random` | `random.seed(semilla)` |
| NumPy | `np.random.seed(semilla)` |
| PyTorch (CPU) | `torch.manual_seed(semilla)` |
| PyTorch (GPU) | `torch.cuda.manual_seed_all(semilla)` |
| cuDNN | `deterministic = True`, `benchmark = False` (reproducibilidad > velocidad) |
| DataLoader | `torch.Generator().manual_seed(semilla)` para el shuffle |
| Augmentación | Determinista por índice (`RandomState(1000 + i)`), independiente de la semilla: misma augmentación para todos los modelos (otra fuente de varianza eliminada) |

La determinización estricta de kernels personalizados no está garantizada por
PyTorch en todas las operaciones; el efecto residual se controla
estadísticamente con las 5 corridas (varianza entre semillas, Sección 6).

### Repetición

5 corridas independientes por modelo (**55 corridas de redes entrenables**:
los 10 modelos + `unet_fuerte`; `empty_mask` y `bbox_mask` no entrenan). Para
nnU-Net, la equivalencia **folds internos ↔
semillas** está declarada y justificada en `docs/NNUNET_PIPELINE.md`
(5 subconjuntos de validación disjuntos dentro del pool de 194 → varianza
comparable). Se reportarán media, desviación estándar e IC 95 % (bootstrap)
por modelo (Sección 6).

### Seguimiento de experimentos

El tracker es el **sistema de archivos versionable**, no un servicio externo:

- `resultados/<modelo>/seed_<s>/resultado.json`: configuración completa
  (`config`), versiones ejecutadas (`entorno`), métricas y curvas — un run
  es autocontenido.
- `resumen_runs.csv`: tabla agregada **derivada** de los JSON (nunca al
  revés; regenerable con `--recomponer-resumen`).
- Reanudabilidad: un run cuenta como hecho solo si su `resultado.json` es
  válido; una sesión de Colab interrumpida se retoma re-ejecutando la misma
  celda.

Justificación frente a MLflow/W&B: en Colab gratuito no hay servidor
persistente ni garantía de sesión; un registro en archivos planos es
auditable, reanudable, se versiona con el código y no añade dependencias de
servicio. Es la herramienta de tracking "otra" que permite la rúbrica.

### Igualdad de cómputo (sin ventajas injustificadas)

| Decisión | Detalle |
|---|---|
| Mismos pasos, no mismo reloj | Todos reciben 150 épocas máximas sobre las mismas 165 imágenes con early stopping idéntico; UNETR (116 M params, tabla 4.3) tarda más por época — se le da el **mismo número de pasos**, no menos |
| Hardware | Misma GPU de sesión de Colab (T4/L4); `entorno.gpu` lo registra por run |
| nnU-Net | Su planificador (parche/batch) es parte del método evaluado; forzarle el batch ajeno lo convertiría en "un U-Net con nombre raro". El coste extra entra al análisis Pareto (Sección 6) |
| nnU-Net | Entrenado con **trainer de 50 épocas** (`entrenamiento_nnunet.py` genera el trainer si el paquete no lo trae), para no heredar el default de 300–1000 épocas |

---

## 4.4 Prevención de data leakage (buenas prácticas obligatorias)

| Regla de la rúbrica | Cumplimiento en este proyecto |
|---|---|
| Separar completamente train/val/test | Splits sellados con SHA-256; verificación automática en cada ejecución |
| Ajustar escaladores solo en train | **No hay escalador ajustable**: z-score por imagen (la normalización de cada imagen no depende de ninguna otra) |
| Sin series temporales | N/A (radiografías estáticas) |
| Fuga por duplicados/partición por grupo | Auditoría EDA A2: 0 duplicados exactos (MD5) y 29 casi-duplicados (pHash) inspeccionados uno a uno; check C6: 0 imágenes fuera del trainval oficial |
| No usar test para hiperparámetros, arquitecturas ni umbrales | El driver no puede leer test (4.1.2); umbral 0,5 y early stopping sobre val; las arquitecturas se fijaron en la Sección 3 antes de resultados |
| Misma validación para todos | Mismos CSV, misma pérdida, mismo bucle, mismas métricas (`metricas_tb.py`) |

---

## 4.5 Evaluación

**Implementación única** en `metricas_tb.py` (si cada modelo midiera con su
propia función, la comparación no sería justa):

| Métrica | Rol | Convenciones fijadas |
|---|---|---|
| Dice | **Primaria** (RQ1–RQ4; $\delta = 0{,}02$) | GT vacío + predicción vacía → 1; GT vacío + predicción no vacía → 0 |
| IoU | Secundaria | Mismas convenciones |
| HD95 (px) | Secundaria (robustez de bordes) | Percentil 95 de la distancia de Hausdorff bidireccional; implementado con numpy+scipy (sin dependencias extra) |
| ASSD (px) | Secundaria | Distancia media simétrica de superficie |
| Dice por estrato | Declarados a priori (EDA A1/A7) | `n_boxes` (1/2/3), `tb_type` (active/latent), área de lesión (pequeña/mediana/grande) |
| Tasa FP en `test_neg` | Control OOD | Nunca se promedia con `test_pos` |

Convención global: HD95/ASSD quedan en NaN cuando alguna máscara es vacía y
se excluyen del promedio **declarando el denominador** (no se imputan).

Lo que la rúbrica 4.5 agrupa bajo "evaluación" y se ejecuta en la fase de
análisis (sus scripts y resultados están en la Sección 7, para no duplicar
texto): calibración de probabilidad por píxel (ECE, Brier), diagnóstico de
sobreajuste/underajuste con pruebas formales, comparación contra el techo
humano, y el análisis cualitativo de errores (visualización predicción vs
GT con `predecir.py` y taxonomía de casos difíciles).

---

## 4.6 Robustez, interpretabilidad y ablación

Las tres pruebas se ejecutan sobre los resultados congelados del benchmark,
en la ruta de la [Sección 7](07_ruta_metodologica.md) (I6–I8). El diseño,
declarado aquí antes de ver resultados:

| Prueba | Diseño |
|---|---|
| Robustez | (a) Control OOD continuo: tasa de FP en `test_neg` (40 sanos, nunca vistos). (b) Perturbaciones controladas aplicadas a `test_pos`: ruido gaussiano, desenfoque y rotación; se reporta la caída de Dice por nivel |
| Interpretabilidad | Grad-CAM sobre los **tres mejores modelos** por Dice-val; coherencia discutida contra la localización de las cajas del anotador (el GT de plata hereda la localización de esas cajas, discusión en Sección 7) |
| Ablación | La propia comparación de los 10 contiene pares-ablación naturales con el **mismo encoder ResNet-34**: U-Net vs Attention U-Net (aporte del módulo de atención) vs U-Net++ (aporte del skip denso). Se suma una corrida complementaria de ablación sobre el mejor modelo: con y sin augmentación (la augmentación es el único componente del protocolo que no es arquitectura) |

Ninguna de estas pruebas reentrena modelos ni modifica los pesos: se calculan
sobre los checkpoints ya registrados, lo que mantiene intacto el presupuesto
del benchmark principal.
