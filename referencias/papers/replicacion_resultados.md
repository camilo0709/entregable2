# Replicación del paper guía — resultados ejecutados

Ejecución de `codigo/replicacion_paper_guia.py` (semilla 42 solo para el muestreo de resolución). Sin entrenamiento: usa únicamente el paquete público liberado de TBX11K.

## 0 · Categorías del paquete (JSON COCO)
- id 1: ActiveTuberculosis
- id 2: ObsoletePulmonaryTuberculosis
- id 3: PulmonaryTuberculosis
- mapeo al paper: 1→active · 2,3→latent

## C1 · Composición train+val (Tabla 2 del paper)
| categoría | paper (train+val) | paquete medido | Δ |
|---|---|---|---|
| healthy | 3800 | 3800 | +0 |
| sick & non-TB | 3800 | 3800 | +0 |
| TB total (carpeta imgs/tb) | 800 | 800 | +0 |
| TB con cajas (json trainval) | 800 | 799 | -1 |
|   active TB | 630 | 630 | +0 |
|   latent TB | 140 | 139 | -1 |
|   active & latent | 30 | 30 | +0 |
|   sin cajas en json | 0 | 0 | +0 |
| TOTAL train+val (carpetas) | 8400 | 8400 | +0 |

## C2 · Partición oficial train (Tabla 2 del paper)
- lista `TBX11K_train.txt`: **6600** líneas (paper: 6 600) → Δ = +0
- por carpeta: tb 600 (paper 600) · sick 3000 (paper 3 000) · health 3000 (paper 3 000)
- clases TB del json de train: active 473 (paper 473) · latent 103 (paper 104) · both 23 (paper 23)

## C3 · Partición val derivada (trainval − train)
- TB en val derivado: **200** (paper: 200)
- sick en val: 800 (paper 800) · health en val: 800 (paper 800)
- clases TB en val: active 157 (paper 157) · latent 36 (paper 36) · both 7 (paper 7)

## C4 · Estadísticas de cajas (Figura 1 del paper, escala 512×512)
- **train (json solo-TB)**: 902 cajas · área/fracción mediana 0.0556 · IQR [0.0340, 0.0808] · p90 0.1116
- **train+val (json solo-TB)**: 1211 cajas · área/fracción mediana 0.0552 · IQR [0.0341, 0.0804] · p90 0.1118
- imágenes TB trainval con ≥1 caja: **799** de 799
- cajas por imagen (1/2/3/≥4): 417/355/24/3

## C5 · Condición de entrada 512×512 (Sec. 4.1 del paper)
- metadatos JSON: width/height ∈ [512]
- muestreo PIL de 50 PNG TB: tamaños observados [(512, 512)]

## C6 · Protocolo: nuestros splits dentro del trainval oficial
- `train.csv` (n=165): fuera del trainval oficial: **0** 
- `val.csv` (n=29): fuera del trainval oficial: **0** 
- `test_pos.csv` (n=80): fuera del trainval oficial: **0** 
- `test_neg.csv` (n=40): fuera del trainval oficial: **0** 
- `test_pos_reserva.csv` (n=64): fuera del trainval oficial: **0** 
- veredicto: ✅ ningún split usa el test oculto del paper

## Discrepancias detectadas (paper vs paquete liberado)
1. `TBX11K_train_only_tb.json` declara **599** imágenes TB (paper: 600) y `TBX11K_trainval_only_tb.json` **799** (paper: 800), mientras las carpetas contienen 600/800 PNG. Causa probable: omisión de una imagen sin cajas al generar los JSON. No afecta a este proyecto (C6).
2. El paquete solo incluye `TBX11K_train.txt` y `all_train.txt`; el README describe también `TBX11K_val.txt`/`TBX11K_trainval.txt`, ausentes. Causa: re-empaquetado de terceros (distribución vía Kaggle). Val reconstruido por diferencia de conjuntos (C3).
3. Resolución: el paper describe los radiografías originales ≈3000×3000, pero la condición de red es 512×512 (Sec. 4.1) y el paquete libera 512×512 (C5). Nuestra entrada nativa 512×512 es equivalente a la suya.
4. Tablas 3–4 (clasificación/detección sobre test): **no reproducibles** — el GT del test de 2 800 no se publicó (declarado por los autores). Este trabajo replica los resultados de *datos* y la condición de entrada; los detectores SSD/RetinaNet/Faster R-CNN/FCOS quedan fuera del Top-10 por ser detección de cajas, no segmentación píxel-a-píxel.
