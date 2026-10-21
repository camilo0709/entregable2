# DECISIÓN D1 — GROUND TRUTH (REGISTRO)

- **Estado:** ✅ **CONFIRMADA** por el usuario
- **Fecha de decisión:** 2026-10-20
- **Decidido por:** Camilo Andrés Mujica Escorcia
- **Ejecutada por:** asistente (análisis + generación de split)

---

## 1. Pregunta de decisión

> ¿Qué se usa como **ground truth (GT)** para calcular Dice, IoU, HD95 y ASSD
> en el benchmark de 10 modelos del Entregable 2?

---

## 2. Evidencia recogida (exploración de `datos/kaggle_dataset/TBX11K`)

### 2.1 TBX11K **no** contiene máscaras de segmentación

| Verificación | Resultado |
|---|---|
| Campo `<segmented>` en XML | `0` en todos los ejemplos revisados |
| Campo `segmentation` en COCO JSON | **ausente** en las 1.211 anotaciones |
| Tipo de anotación | `bndbox` (bounding box rectangular) |
| Anotaciones totales | 1.211 boxes en 799 imágenes |
| Categorías | `ActiveTuberculosis` (972), `ObsoletePulmonaryTuberculosis` (239), `PulmonaryTuberculosis` (0) |
| Fracción de imagen del box | mediana 5,3 % (min 0,3 %, max 25,8 %) |

**Conclusión parcial:** TBX11K es un dataset de **localización/detección**, no de
segmentación. No puede sustentar métricas píxel-a-píxel directamente.

### 2.2 El GT actual del proyecto es circular

El `npz_dataset` (194 imágenes) contiene dos campos:

| Campo | Origen real | ¿Sirve como GT? |
|---|---|---|
| `bbox_mask` | Rectángulo del box de TBX11K | ❌ etiqueta débil |
| `mask` | "Máscara **Refinada (SAM 2)**" (`visualizar_npz.py:151`) | ❌ pseudolabel de SAM 2 |

Medida de circularidad: **Dice(`mask`, `bbox_mask`) = 0,773** (mediana 0,783),
y `mask` tiene 1,21× el área del box → SAM 2 hereda el prior de la caja.

> **Razonamiento clave:** si el GT es salida de SAM 2, medir un modelo contra él
> solo mide "qué tan bien imita a SAM 2", no "qué tan bien segmenta TB".

### 2.3 Disponibilidad de imágenes

| Pool | Descripción | N |
|---|---|---|
| **A** | NPZ con pseudomáscara SAM 2 (`train`+`val` actuales) | 194 |
| **B** | Imágenes `tb/` en disco **con box y sin usar** en NPZ | 144 |
| **C** | Imágenes sanas `health/` en disco | 1.380 |
| — | TBX11K con box en disco | 195 de 799 |
| — | TBX11K con box **ausente** en disco | 604 (copia Kaggle incompleta) |

### 2.4 Comprobación anti-fuga (ejecutada)

| Prueba | Resultado |
|---|---|
| Duplicados exactos (MD5) entre splits | **0** |
| Casi-duplicados (dHash ≤4 bits) | 48 pares |
| De esos, correlación pixel > 0,99 | **0** |
| Pares train/val con corr > 0,95 | **0** |
| Correlación máxima observada | 0,91 (anatomía similar, no duplicado) |
| IDs repetidos entre splits | **0** |
| Duplicados exactos en TBX11K que cruzan `test/`↔`sick/` | 13 (afecta a negativos, no a nuestro test) |

**Veredicto:** no hay fuga por imagen duplicada. *Limitación declarada:* TBX11K
**no expone ID de paciente**, por lo que la partición "por paciente" no es
verificable con los metadatos disponibles. Se compensa con detección de
duplicados/casi-duplicados (arriba) y se declara como limitación.

---

## 3. Decisión

**D1 = Test set híbrido de 3 niveles.**

| Nivel | Pool | GT | Uso |
|---|---|---|---|
| **Train/Val** | A (194 NPZ) | Pseudomáscaras SAM 2 | Entrenar los 10 modelos — etiqueta débil **aceptable solo aquí** |
| **Test primario** | B (80 de 144) | **Anotación manual humana** | Base legítima para Dice/IoU/HD95/ASSD y Tablas A/B/C |
| **Test secundario** | C (40 sanos) | Máscara vacía por construcción | Especificidad / falsos positivos |
| Reserva | B (64 restantes) | — | Regeneración / anotación extra si hace falta |

### Por qué esta opción y no otra

| Alternativa | Por qué se descarta |
|---|---|
| GT = pseudomáscaras SAM 2 | Circular; imposible reclamar precisión clínica real |
| GT = boxes de TBX11K | Convierte Dice en "overlap con rectángulo"; no es segmentación |
| Reformular a solo destilación | Pierde el enfoque de segmentación clínica y el sentido del benchmark |
| **Test set manual híbrido** ✅ | Única vía con GT real verificable; además encaja con el gap de investigación |

---

## 4. Consecuencias para el Entregable 2

1. **Trabajo humano requerido:** anotar 80 imágenes positivas (trazar la lesión
   píxel a píxel). Estimado: 3–6 min/imagen → **4–8 horas**.
2. **El test se sella** con `datos/splits/split_manifest.json` (`"sealed": true`)
   antes de cualquier experimento.
3. **Gap de investigación reforzado:** entrenar con etiquetas débiles (destilación
   de SAM 2) y demostrar con test experto que un modelo ligero iguala o supera a
   su teacher, con calibración. Coherente, defendible y publicable.
4. **D3 (subtask):** `tb_type` del pool A = 159 `active_tb` / 35 `latent_tb`
   → subgrupo secundario viable pero pequeño; se reporta como exploratorio.

---

## 5. Artefactos generados

| Archivo | Contenido |
|---|---|
| `codigo/generar_split_entregable2.py` | Generador reproducible del split (semilla 42) |
| `datos/splits/train.csv` | 165 imágenes pool A |
| `datos/splits/val.csv` | 29 imágenes pool A |
| `datos/splits/test_pos.csv` | **80 imágenes — GT manual pendiente** |
| `datos/splits/test_neg.csv` | 40 sanos — máscara vacía |
| `datos/splits/test_pos_reserva.csv` | 64 imágenes de reserva |
| `datos/splits/split_manifest.json` | Semillas, conteos, comprobaciones, hashes |

---

## 6. Pendiente derivado de D1

- [ ] **Anotar manualmente** las 80 imágenes de `test_pos.csv`
- [ ] Definir protocolo de anotación (ver `PROTOCOLO_ANOTACION.md`)
- [ ] Definir inter-anotador / acuerdo (mínimo doble anotación en submuestra)
- [ ] **D2** — cómputo/GPU (sin resolver)
- [ ] Recuperar `pipeline_segmentacion_tb.py`, `validar_datos.py`,
      `ejemplo_train_unet.py` y docs faltantes desde Drive

---

## R1. REVISIÓN D1-R1 — GT DEL TEST: DE MANUAL A PLATA (SILVER STANDARD)

- **Estado:** ✅ **APROBADA** por el usuario (Camilo Andrés Mujica Escorcia)
- **Fecha de la revisión:** 2026-10-20
- **Ejecutada por:** asistente (documentación + pipeline congelado)
- **Momento:** ANTES de fijar la fecha del preregistro
  (`sellado_fecha_declarada` seguía sin fijar) y ANTES de entrenar cualquier
  modelo del benchmark → enmienda de protocolo legítima, no snooping.

### R1.1 ¿Qué cambia?

| | D1 original | D1-R1 |
|---|---|---|
| GT de `test_pos` (80 imgs) | anotación manual humana | **pseudomáscara SAM 2** (plata) |
| GT de `train`/`val` (194) | pseudomáscara SAM 2 | pseudomáscara SAM 2 (regenerada homogéneamente) |
| Splits (ids, conteos, hashes) | sellados | **idénticos, intactos** |
| Circularity mitigations | test experto separa teacher de estudiantes | validación humana de submuestra (ver R1.3) |

### R1.2 Motivo

TBX11K no contiene máscaras de segmentación (solo bounding boxes; §2.1). La
anotación manual de 80 lesiones píxel a píxel requiere criterio radiológico
que el usuario no posee; una anotación sin expertise produciría un GT
cuestionable a un costo alto (4–8 h) y sin garantía de calidad. Se decide
usar el mismo pipeline de refinamiento (SAM 2 con prompt de caja) para
generar el GT del test, **declarándolo explícitamente como silver standard**.

### R1.3 Salvaguardas (obligatorias, ninguna opcional)

1. **Teacher externo:** SAM 2 (Meta) no fue entrenado ni afinado con TBX11K;
   cero exposición a las imágenes del benchmark → no hay fuga de etiquetas
   desde el teacher. El único puente teacher↔datos es la caja del anotador
   original de TBX11K (etiqueta débil).
2. **Validación humana de submuestra (techo de desempeño):** el usuario
   corrige manualmente (con visor simple, sobre la máscara SAM 2 pre-cargada)
   una submuestra aleatoria estratificada de **20 imágenes de `test_pos`**
   (15–20). Se calcula Dice(humano, SAM 2) como acuerdo inter- método; ese
   valor se reporta como **techo de desempeño** del benchmark. Protocolo en
   `PROTOCOLO_ANOTACION.md` (§ ajuste R1); validación con
   `codigo/validar_ground_truth_test.py --anotador-b <dir>`.
3. **Congelado previo:** el GT plata se genera con UN solo script
   (`codigo/generar_pseudo_gt.py`) y UN solo checkpoint
   (`sam2_hiera_tiny.pt`, SHA-256 registrado) ANTES de entrenar cualquier
   modelo del benchmark. Todo queda en `datos/pseudo_gt_manifest_v2.json`.
4. **Declaración obligatoria en el informe:** el Jupyter Book y el Data Card
   declararán que el GT es plata y que las métricas del benchmark miden
   **imitación del pipeline teacher** (con sus sesgos heredados de la caja),
   no calidad clínica absoluta; los modelos que "superen" al teacher
   simplemente podrían estar sobre-ajustando a sus sesgos.

### R1.4 Qué NO cambia

- Los 6 CSV de `datos/splits/` y su `split_manifest.json` original: **byte a
  byte idénticos** (SHA-256 verificables con
  `python codigo/verificar_integridad_splits.py`). Las rutas Windows se
  remapean solo al cargar (`dataset_tb._ruta_local`).
- El diseño experimental: 10 modelos × 5 semillas, presupuesto idéntico,
  métricas, baselines, tests estadísticos.
- `test_neg` (40 sanos): máscara vacía por construcción, sin cambio.

### R1.5 Artefactos de esta revisión

| Archivo | Contenido |
|---|---|
| `codigo/generar_pseudo_gt.py` | Pipeline congelado del GT plata (SAM 2 bbox→mask) |
| `datos/pseudo_gt_manifest_v2.json` | Manifiesto: hashes, parámetros, conteos, fecha |
| `codigo/verificar_integridad_splits.py` | Verifica hashes sellados vs manifest |
| `datos/ground_truth_test/<id>.png` | GT plata del test_pos (0/255, 512×512) |
| `datos/npz_dataset/**.npz` | Pool A regenerado homogéneamente (v1 respaldado) |

### R1.6 Hallazgo: los hashes del manifest v1 no verifican

Al restaurar los CSV desde Google Drive se detectó que los SHA-256 de
`split_manifest.json` (v1) **no corresponden a ninguno de los CSV**
recuperados (se comprobó también normalizando CRLF→LF, sin coincidencia).
El desfase existía **ya en la propia fuente de Drive** (los archivos
descargados frescos de ahí tampoco verificaban), por lo que no es un efecto
de la restauración local.

Lo que SÍ se verificó y se cumple: conteos exactos (165/29/80/40/64),
disjunción total entre splits, sin ids repetidos, `sealed: true`.

**Actuación:** se crea `datos/splits/split_manifest_v2.json` como **línea
base vigente** (hashes actuales + los v1 declarados como históricos), sellado
antes de cualquier entrenamiento. `verificar_integridad_splits.py` exige
v2 y reporta v1 como histórico. Esta discrepancia se declara en el Data Card
y en la sección de reproducibilidad del book.

### R1.7 Columna `gt_tipo` de los CSV: valor histórico, no vigente

Los CSV sellados de `test_pos` traen `gt_tipo=manual_pendiente`, etiqueta de
la fase previa a D1-R1. Los CSV **no se modifican jamás** (integridad byte a
byte sellada en `split_manifest_v2.json`), por lo que esa columna permanece
como **metadato histórico**. La GT vigente del test es la plata (SAM 2) y lo
son, en orden de autoridad: esta decisión (§R1.1),
`datos/pseudo_gt_manifest_v2.json` y `ground_truth_test/ANOTACION_METRICAS.json`.
Cualquier herramienta o informe que lea `gt_tipo` debe tratarlo como rótulo
de origen, no como descripción de la GT actual.

### R1.8 Hallazgo: la columna `ruta` de los CSV está cruzada respecto al split

La columna `ruta` de los CSV sellados apunta al directorio opuesto al split
en parte de las filas:

| CSV | filas | `ruta` → `npz_dataset/train/` | `ruta` → `npz_dataset/val/` |
|---|---|---|---|
| `train.csv` | 165 | 123 | **42** |
| `val.csv` | 29 | **20** | 9 |

Por eso el listado de directorios da 143 archivos en `train/` y 51 en
`val/` (total 194, correcto) mientras que la **membresía** real del split es
165/29.

**Cómo se resuelve (sin tocar los CSV):** tanto el loader
(`dataset_tb.__getitem__`, que usa `fila["ruta"]` vía `_ruta_local`) como el
generador del GT (`generar_pseudo_gt.py`, que escribe en la ruta dictada por
`fila["ruta"]`) siguen la misma referencia, de modo que las 194 filas
encuentran exactamente su npz (verificado 194/194 con metadata
`label`/`tb_type`/`fname` correcta).

**Regla derivada (obligatoria para las herramientas posteriores):** la
membresía de un split la define **el CSV (columna `id`)**, nunca el nombre
del directorio. Ningún script puede enumerar `npz_dataset/train/` para
decidir qué datos son de entrenamiento — eso mezclaría 42 imágenes de train
y 20 de val. Declarar esta peculiaridad en el Data Card y en la sección de
reproducibilidad del book.
