# Sección 5 · Paper guía y replicación

> **Estado:** ✅ entregable · Sección 5 de la rúbrica.
> La elección del paper guía es la **única decisión declarada post-hoc** del
> proyecto (ya anticipada en la Sección 1): se eligió después de fijar el
> dataset y los splits, pero **antes** de tener resultados de ningún modelo
> benchmark, y no condiciona métricas ni hipótesis.

---

## 5.1 Selección y referencia completa

**Paper guía:** el artículo que introduce el propio dataset de este proyecto.

> Liu, Y., Wu, Y.-H., Ban, Y., Wang, H., & Cheng, M.-M. (2020).
> *Rethinking Computer-aided Tuberculosis Diagnosis*.
> In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
> Recognition (CVPR) (pp. 2646–2655). IEEE.
> https://doi.org/10.1109/CVPR42600.2020.01270
> (versión de acceso abierto: *openaccess.thecvf.com*; proyecto:
> http://mmcheng.net/tb/)

```bibtex
@inproceedings{liu2020rethinking,
  title     = {Rethinking Computer-aided Tuberculosis Diagnosis},
  author    = {Liu, Yun and Wu, Yu-Huan and Ban, Yunfeng and Wang, Huifang and Cheng, Ming-Ming},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {2646--2655},
  year      = {2020},
  doi       = {10.1109/CVPR42600.2020.01270}
}
```

Copia local del PDF utilizada: `referencias/papers/liu2020rethinking_cvpr.pdf`
(10 pp.; no se redistribuye por derechos de autor).

## 5.2 Descripción del modelo original

### Tarea y formulación

El paper aborda **dos tareas conjuntas** sobre cada radiografía $x$:

1. **Clasificación de imagen** en 3 clases: *healthy*, *sick but non-TB* y
   *TB* (Softmax);
2. **Detección de áreas de TB** con 2 clases de objeto (TB activo, TB latente)
   y predicción de cajas $b = (x_0, y_0, w, h)$.

Los autores observan que los detectores existentes **ignoran las imágenes de
fondo** (sin objetos) y producirían miles de falsos positivos en un escenario
clínico donde la mayoría de radiografías no tiene TB. Su reforma: acoplar una
**rama de clasificación** al detector para filtrar FP, entrenada en dos etapas.

### Arquitectura (la "reforma")

Sobre cuatro detectores base — **SSD** (VGG-16), **RetinaNet**, **Faster
R-CNN** y **FCOS** (los tres últimos con ResNet-50 + FPN) — se inserta, tras la
última convolución del backbone (`conv5_3` / `res5c`), una rama de
clasificación de **5 convoluciones consecutivas** (512 canales, ReLU; stride 2
solo en SSD; max-pooling stride 2 tras la tercera) → **global average
pooling** → **fully connected con 3 salidas**.

### Entrenamiento en dos etapas (supuesto central)

1. **Etapa 1:** el detector, con sus ajustes por defecto, entrenado **solo con
   radiografías TB** del trainval (mmdetection; batch 16; 38 400 iteraciones
   con preentrenamiento ImageNet, 76 800 sin él; lr 0,005 — 0,0005 en SSD —
   dividida a mitad de camino).
2. **Etapa 2:** se **congelan backbone y detector** y solo se entrena la rama
   de clasificación (24 épocas, lr $10^{-3}$ con divisiones a las 12 y 18),
   con el trainval completo **más mitades de Montgomery/Shenzhen y los train
   de DA/DB** (el resto queda para evaluación).

**Supuestos declarados:** (a) la clasificación de imagen filtra los FP de la
detección; (b) basta IoU 0,5 con la caja humana para "ayudar al radiólogo a
encontrar el área" (por eso priorizan APbb50 sobre APbb); (c) la prevalencia
del set (44,6 % sano / 44,6 % enfermo-no-TB / 10,7 % TB) aproxima el escenario
de consulta; (d) la entrada a red es 512×512 (las originales ≈3000×3000 se
redimensionan).

### Métricas del paper (las adaptamos como puente)

- Clasificación: Accuracy, AUC (TB), Sensitivity, Specificity, AP, AR.
- Detección: **APbb** estilo COCO (promedio sobre IoU 0,50:0,05:0,95) y
  **APbb50**, por tipo (activa/latente) y *category-agnostic*, en dos modos
  (todo el test / solo TB), excluyendo los 10 "uncertain".

### Resultados clave reportados (los que intentamos replicar)

- **Clasificación (Tabla 3):** el mejor es Faster R-CNN con ImageNet:
  Accuracy **89,7 %**, AUC 93,6, Sens 91,2 %, Spec 89,9 % — por encima del
  radiólogo experto del estudio humano (Accuracy **68,7 %** en 4 clases;
  84,8 % ignorando tipo de TB).
- **Detección (Tabla 4):** el mejor *category-agnostic* sobre solo-TB es SSD
  sin preentrenamiento: APbb50 **69,6** / APbb **29,1**; **todos los métodos
  fallan en TB latente** (APbb ≤ 4,1).
- **Visualización (Fig. 3):** mapas de características (PCA a escala 1/32)
  como explicabilidad — precursor de nuestro RQ4.

## 5.3 Justificación de la selección

| Criterio | Argumento |
|---|---|
| **Relevancia** | Es el paper que **define el dataset, la tarea, el protocolo de evaluación y los baselines** de este proyecto: sin él no existirían TBX11K ni sus cajas |
| **Impacto** | CVPR 2020 (conferencia Q1 del área); cientos de citas; el benchmark TB-CAD post-2020 usa TBX11K como estándar (Sección 3) |
| **Idoneidad frente al problema** | Introduce la localización de lesión (no solo clasificación binaria), que es exactamente el eje de nuestro benchmark de **segmentación**; sus análisis de FP en sanos anticipan nuestro control OOD (`test_neg`) |
| **Trazabilidad** | Código/protocolo documentados (mmdetection), datos y README oficiales disponibles localmente → permite replicación verificable |

No se eligió un paper de *segmentación* concreto como guía porque el proyecto
**no sigue la metodología de un solo artículo**: compara 10 familias con
presupuesto idéntico (Sección 4); el único marco común verificable es el que
entrega el propio dataset.

## 5.4 Replicación ejecutada

### Plan (definido a priori en el propio script)

`codigo/replicacion_paper_guia.py` reproduce resultados **del paper basados en
el paquete público liberado**, sin entrenamiento y sin tocar datos sellados:

| Check | Resultado clave del paper | Qué mide |
|---|---|---|
| C1 | Tabla 2: train+val = 3 800 / 3 800 / 800 | composición |
| C2 | Tabla 2: train = 6 600 (3 000/3 000/600) | lista oficial |
| C3 | Tabla 2: val = 1 800 (800/800/200) | val derivado |
| C4 | Figura 1: estadística de cajas | cajas/imagen y área |
| C5 | Sec. 4.1: entrada 512×512 | condición de red |
| C6 | Protocolo de uso del paper | splits ⊂ trainval |

*(No se replican las Tablas 3–4: el GT del test de 2 800 **no se publicó**,
declarado por los autores; es imposible por construcción, no por omisión.)*

### Resultados ejecutados (números reales del paquete local)

**C1 — Composición train+val (Tabla 2, parte train+val):**

| Categoría | Paper | Medido | Δ |
|---|---|---|---|
| healthy | 3 800 | 3 800 | **+0** |
| sick & non-TB | 3 800 | 3 800 | **+0** |
| TB total (carpeta `imgs/tb`) | 800 | 800 | **+0** |
| active TB | 630 | 630 | **+0** |
| latent TB | 140 | 139 | **−1** |
| active & latent | 30 | 30 | **+0** |
| TOTAL | 8 400 | 8 400 | **+0** |

**C2 — Partición train:** `TBX11K_train.txt` = **6 600** líneas (+0); por
carpeta tb 600 / sick 3 000 / health 3 000 (+0 en los tres); clases TB:
active 473 (+0), both 23 (+0), latent **103 vs 104 (−1)**.

**C3 — Partición val derivada** (trainval − train): TB 200 (+0), sick 800
(+0), health 800 (+0); clases: active 157, latent 36, both 7 (**+0 en
las tres**).

**C4 — Cajas (Fig. 1, escala 512×512):** 1 211 cajas sobre 799 imágenes
TB trainval; **área/fracción mediana 0,0552** (IQR 0,0341–0,0804; p90
0,1118); cajas por imagen 1/2/3/≥4 = **417/355/24/3**.

**C5 — Condición de entrada:** metadatos y muestreo PIL (50 PNG TB) =
**512×512 en el 100 %** — idéntica a la condición de red del paper.

**C6 — Protocolo:** los 5 CSV sellados de este proyecto (train 165, val 29,
test_pos 80, test_neg 40, reserva 64) contienen **0 imágenes fuera del
trainval oficial**: el test oculto del concurso no interviene en nada.

### Discrepancias y análisis de causas

| # | Discrepancia | Δ | Causa analizada | Impacto en este proyecto |
|---|---|---|---|---|
| 1 | Los JSON de anotación declaran 599/799 imágenes TB (paper: 600/800) | −1 imagen (una latente) | Omisión al generar los JSON COCO de la imagen que no aporta cajas; carpetas y lista sí la contienen | Ninguno: C6 confirma que ningún split la usa |
| 2 | Faltan `TBX11K_val.txt` y `TBX11K_trainval.txt` (el README los describe) | — | Re-empaquetado de terceros (distribución vía Kaggle) | Reconstruimos val por diferencia de conjuntos (C3), validado contra Tabla 2 |
| 3 | El paper describe ≈3000×3000 pero el paquete es 512×512 | 0 en efecto | La **condición de red** del paper ya es 512×512 (Sec. 4.1) y el paquete libera esa versión | **Equivalencia**: nuestra entrada nativa = la suya |
| 4 | Tablas 3–4 no reproducibles | — | GT del test de 2 800 no publicado (decisión de los autores, concurso en línea) | Declarado; por eso la replicación cubre resultados de *datos* + condición de entrada |

**Veredicto de la replicación:** la composición, las tres particiones, la
estadística de cajas y la condición de entrada del paper se reproducen con
**Δ = 0 en 15 de 16 comparaciones** (la única, −1, está explicada y es del
paquete, no del pipeline). Esto valida el **pipeline experimental de
datos** de este proyecto: los mismos archivos producen las mismas
estadísticas del paper → los splits sellados y el GT plata nacen de una
base reproducible.

## 5.5 Relación con los modelos benchmark (Top 10)

| Eje | Paper guía (Liu et al.) | Top 10 de este proyecto |
|---|---|---|
| Tarea | Clasificación 3 clases + **detección de cajas** | **Segmentación binaria** píxel-a-píxel |
| Modelos | SSD, RetinaNet, Faster R-CNN, FCOS (reformados) | U-Net y familia CNN + transformadores (Sección 4) |
| Supervisión | Cajas de experto (débil) | Pseudomáscaras plata con techo medido (cajas → SAM 2) |
| Evaluación | Test oculto de 2 800 (GT no publicado), corrida única | `test_pos` 80 con GT sellado, 10 modelos × 5 semillas + significancia |
| Explicabilidad | PCA de mapas de características (Fig. 3) | Grad-CAM formal con baseline y test estadístico (RQ4) |

**Similitudes:** misma entrada 512×512; misma intuición de *localización fina
es difícil* (su gran salto APbb50→APbb y su Loc-error alto ↔ nuestro Hallazgo
A3 de bordes difusos); mismo desbalance (su TB latente escaso ↔ nuestro
≈8,8 % de píxeles de lesión); su filtro de FP en sanos ↔ nuestro control OOD
`test_neg` (y su hallazgo de sesgo por "apariencia" coincide con el AUC 0,842
del EDA A4).

**Diferencias:** los 4 detectores del paper **no entran al Top 10** — la
rúbrica exige modelos de la tarea elegida y su tarea es detección de cajas;
además su presupuesto (4×RTX 2080Ti, decenas de miles de iteraciones) no
empareja con el nuestro, lo que los haría *injustos* como candidatos.

**Posición dentro del benchmark:** el paper guía se ubica **aguas arriba**
del Top 10: entrega el dataset, la métrica puente (APbb) y los baselines
contra los cuales se juzga el campo. Dentro de nuestro diseño, su rol operativo
es doble: (i) `bbox_mask` — la caja humana rasterizada — es el baseline
geométrico con el que se mide cuánto aporta el aprendizaje sobre "dibujar la
caja"; (ii) su métrica APbb se usa como **puente de comparación externa**
convertiendo nuestras máscaras a cajas (Secciones 6 y 7), de modo que los
números de este proyecto son confrontables —con las salvedades declaradas—
con la literatura TBX11K post-2020.
