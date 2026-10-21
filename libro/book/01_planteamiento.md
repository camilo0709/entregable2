# Sección 1 · Definición del problema, preguntas de investigación y dataset


---

## 1.1 Definición formal del problema

### Formulación matemática

- **Entrada** $x$: radiografía de tórax en grises, $x \in [0,1]^{1 \times 512 \times 512}$ (1 canal, escala [0,1], sin reescalado).
- **Salida** $y$: máscara binaria de lesión por tuberculosis,
  $y \in \{0,1\}^{512 \times 512}$ (píxel de lesión = 1, resto = 0).
- **Espacio de hipótesis** $\mathcal{H} = \{f_\theta : \mathbb{R}^{1\times512\times512} \to [0,1]^{512\times512}\}$,
  una familia de 10 arquitecturas benchmark (Sección 4) entrenadas con la misma
  pérdida combinada
  $$\mathcal{L}(\theta) = \underbrace{\tfrac{1}{HW}\sum_i \mathrm{BCE}\big(f_\theta(x)_i,\, y_i\big)}_{\text{entropía cruzada binaria}}
    + \underbrace{1 - \tfrac{2\sum_i f_\theta(x)_i\, y_i}{\sum_i f_\theta(x)_i + \sum_i y_i + \varepsilon}}_{\text{Dice loss}}$$
  idéntica pesos para los 10 modelos (decisión A1 del EDA, Sección 2).
- **Función objetivo** (entrenamiento, solo train):
  $$\theta^{*} = \arg\min_{\theta \in \mathcal{H}} \; \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{train}}} \big[\mathcal{L}(\theta)\big],
  \qquad \hat{y} = \mathbb{1}\big[f_{\theta^{*}}(x) \ge 0.5\big]$$
  umbral de inferencia fijo en 0.5 para los 10 modelos (sin calibración por modelo).

### Tipo de tarea

**Segmentación semántica binaria** (no clasificación ni detección): la salida es
píxel-a-píxel sobre la imagen completa. Las cajas delimitadoras originales de
TBX11K solo se usan como *baseline* geométrico (`bbox_mask`) y como referencia
para Grad-CAM (RQ4), nunca como objetivo de entrenamiento.

### Relevancia científica y aplicada

- **Científica:** la literatura de segmentación de TB compara arquitecturas con
  corrida única, presupuestos desiguales y sin pruebas de significancia
  (9/10 sin varianza, 10/10 sin tests; Sección 3.4). Este problema formula un
  benchmark **estadísticamente contrastable**: 10 modelos × 5 semillas,
  presupuesto idéntico, GT con techo humano medido y control OOD — cierra el
  gap documentado en la Sección 3.
- **Aplicada:** el tamizaje de TB con radiografía es la vía de cribaje de mayor
  alcance en países de alta carga, con escasez de radiólogos. Una segmentación
  confiable (con límites explícitos) sirve para **priorizar estudio**, no para
  diagnóstico autónomo; la explicabilidad (RQ4) es requisito de adopción
  clínica, no adorno.

### Métricas declaradas a priori

| Rol | Métrica | Detalle |
|---|---|---|
| **Primaria** | **Dice** en `test_pos` | media por imagen sobre los 80 casos; intervalos por bootstrap IC 95 % |
| Secundaria 1 | IoU | misma base que Dice, escala distinta |
| Secundaria 2 | HD95 (px) | robustez de bordes; comparable sin resize (A5) |
| Secundaria 3 | ASSD (px) | distancia promedio de superficie |
| Estratos | Dice por `n_boxes` (1/2/3), `tb_type` (active/latent) y área (pequeña/mediana/grande) | declarados antes de entrenar (EDA A1, A7) |
| Control OOD | Tasa de FP en `test_neg` (40 sanos) | **nunca** se promedia con `test_pos` (EDA A4) |

**Margen de relevancia práctica:** $\boldsymbol{\delta = 0.02}$ de Dice absoluto
(2 puntos porcentuales). Justificación: diferencias de 1–2 pp entre variantes de
U-Net son habituales en la literatura de TB y caen dentro de la dispersión típica
entre semillas; un efecto solo se declara *prácticamente relevante* si
$\text{IC}_{95\%}$ de la diferencia no contiene a δ **y** $p < 0.05$.
Diferencias $< \delta$ se reportan como empate práctico aunque sean significativas.

---

## 1.2 Preguntas de investigación e hipótesis

| # | Pregunta de investigación (contrastable) | H₀ | H₁ | Prueba estadística |
|---|---|---|---|---|
| **RQ1** | ¿Existen diferencias significativas entre las 10 arquitecturas (y sus familias) al segmentar lesiones TB bajo presupuesto idéntico? | Todos los modelos tienen la misma distribución de Dice en `test_pos` (rangos de Friedman iguales) | Al menos un modelo difiere | **Friedman** (α = 0,05) sobre las 5 semillas × 10 modelos; si se rechaza: **Nemenyi** (CD) + **Wilcoxon pareado por imagen con Holm** |
| **RQ2** | ¿Los transformadores/híbridos superan a las CNN clásicas en baja señal (lesión ≈ 8 % px) o solo empatan a mayor coste? | La mediana de las diferencias de Dice (grupo transformador/híbrido − grupo CNN), emparejada por semilla e imagen, es 0 | La mediana difiere de 0 (y $\lvert\Delta\rvert \ge \delta$ para afirmar superioridad) | **Wilcoxon por rangos con signo** por par de modelos (80 imágenes) + **Holm**; agregación por familia con la misma prueba sobre 5 semillas |
| **RQ3** | ¿Cuánto del error observado se debe al techo del GT (ruido de la anotación plata) y cuánto al modelo? | El mejor modelo ya está a ≤ δ del techo humano (0,9879): $\text{techo} - \text{Dice}_{\text{mejor}} \le \delta$ | El mejor modelo queda por debajo del techo en más de δ | **Bootstrap** (10 000 remuestreos de imágenes) del IC 95 % de la diferencia contra δ; complemento: correlación error vs acuerdo humano por imagen |
| **RQ4** | ¿Grad-CAM coincide con la localización humana de la lesión (¿aciertan por las razones equivocadas)? | La intersección de la máscara top-k de Grad-CAM con la caja/máscara GT **no supera** a los baselines (aleatorio y `bbox_mask`) | La intersección supera a los baselines | **Wilcoxon pareado por imagen** (solapamiento CAM vs baseline, top-3 modelos) + **Holm** |

Notas de contraste:

- **RQ1 es global**; solo si se rechaza H₀ se abren los contrastes por pares
  (esto evita inflar el error tipo I).
- Todas las decisiones de umbral usan α = 0,05 corregido (Holm) y el criterio
  doble significancia + δ de la Sección 1.1.
- La batería completa (normalidad, Friedman + Iman-Davenport, Nemenyi/Wilcoxon-Holm,
  bootstrap, efecto) se ejecuta en la Sección 6.

---

## 1.3 Justificación del dataset

### Criterios de selección

| Criterio | Decisión → evidencia |
|---|---|
| **Relevancia** | TBX11K es *el* corpus público con cajas de lesión TB en CXR de tamaño decente; alineado con el problema (lesión, no diagnóstico binario) |
| **Tamaño** | 11 200 imágenes 512×512; el subconjunto con TB positivo y anotación usable (274 casos con máscara plata + 64 de reserva) permite 5-fold/5-semillas con test de 80 |
| **Calidad** | Anotación por expertos (cajas) en el origen; resolución nativa sin resize; sin duplicados detectados (EDA A2: 0 MD5, 29 pHash auditados → pacientes distintos) |
| **Uso previo en literatura Q1/CVR** | Presentado en **CVPR 2020** (Liu et al.) y decenas de trabajos posteriores de TB-CXR (Revisión sistemática, Sección 3, Flujo A) |

### Licencia, procedencia, ética y privacidad

- **Procedencia** (README oficial del paquete): TBX11K agrega cuatro conjuntos
  públicos — DA (156), DB (150), Montgomery (138, Jaeger et al. 2014) y
  Shenzhen (662, Jaeger et al. 2014) — más 2 800 radiografías propias de test
  (concurso en línea; su GT **no** está publicado). Split oficial:
  6 600 train / 1 800 val / 2 800 test.
- **Licencia:** el paquete **no declara licencia formal**; exige solo citación
  (BibTeX `liu2020rethinking` en el README). Uso académico/no comercial con
  atribución; los conjuntos subyacentes (Jaeger 2014; Chauhan 2014) se
  publicaron abiertamente con fines de investigación. Se declara la ausencia de
  licencia explícita en lugar de asumirla.
- **Ética y privacidad:** radiografías de-identificadas; el paquete no incluye
  datos demográficos identificables ni nombres de paciente (de hecho TBX11K
  **no expone ID de paciente** — limitación arriba). No hay uso clínico
  autorizado: es investigación; las imágenes no se redistribuyen en el repo
  (solo splits, hashes y código — `.gitignore` excluye las imágenes).
- **Consentimiento:** los conjuntos originales se recogieron con finalidad de
  investigación asistencial y se publicaron anonimizados por sus autores
  (Jaeger et al. 2014; Chauhan et al. 2014); TBX11K los re-distribuye en
  igual condición.

### Limitaciones conocidas (declaradas a priori)

1. **Sin ID de paciente** ⇒ no hay partición por sujeto garantizada; el control
   es la auditoría perceptual de duplicados (EDA A2).
2. **Sesgo de adquisición** entre sanos y TB confirmado: AUC 0,842 con solo
   estadísticos de apariencia (p-permutación 0,010, EDA A4) ⇒ `test_neg` es
   control OOD, **no** task.
3. **Shift de tamaño de lesión** train→test (0,088 vs 0,073, KS p = 0,037,
   EDA A7) ⇒ reporte por estrato de área.
4. **GT de plata (plata)**: todas las máscaras son pseudomáscaras SAM 2
   (Sección 1.4); el techo humano declarado es 0,9879 — ningún modelo puede
   superarlo de forma interpretable.
5. **2D monocroma, binario**: no cubre multi-clase ni tomografía; los resultados
   no se extrapolan a otras modalidades (Sección 7, ruta I1–I8).

---

## 1.4 Data Card (descripción técnica)

### Tamaño, estructura y tipo de datos

| Elemento | Valor |
|---|---|
| Corpus origen | TBX11K: **11 200** CXR PNG **512×512**, 5 categorías (Healthy, Sick non-TB, Active TB, Latent TB, Uncertain TB), anotación original = **cajas** (XML/JSON COCO) |
| Split oficial origen | 6 600 / 1 800 / 2 800 (test GT no publicado) |
| **Nuestros splits (sellados)** | train **165** · val **29** · `test_pos` **80** · `test_neg` **40** · `test_pos_reserva` **64** (todos TB positivos salvo `test_neg`) |
| Artefactos de entrada | `datos/npz_dataset/` (194 npz de train+val: imagen + máscara) y PNG 512×512 para test |
| Máscaras plata | **274** (train 165 + val 29 + test_pos 80) |
| Tipo | imagen 2D monocroma + máscara binaria; sin metadatos clínicos |

### Variable objetivo y protocolo de etiquetado (con ruido)

- **Objetivo:** $y$ = máscara binaria de lesión TB activa/latente visible
  (infiltrados, cavidades, nódulos, atelectasias, opacaciones dentro de campo
  pulmonar; excluye estructuras no patológicas).
- **Protocolo (cadena de etiquetado):**
  1. Cajas de experto del TBX11K (origen, etiqueta **débil**).
  2. Pseudomáscara con **SAM 2** (`sam2_hiera_tiny.pt`, cfg `sam2_hiera_t.yaml`)
     con pipeline **congelado**: semilla 42, `min_component_px` 32, CPU —
     274 máscaras, 0 errores, 1 723,9 s (`pseudo_gt_manifest_v2.json`).
  3. En `test_pos`: validación 80/80 (formato y área) + **acuerdo humano vs
     plata: techo Dice 0,9879** (mínimo 0,7583, caso tb0393); decisión
     **D1-R1 congelada**: ninguna máscara se reescribe tras el EDA.
- **Ruido de etiquetas esperado y cuantificado:** ruido **débil** (caja →
  frontera inferida por modelo), con **techo medido** en vez de supuesto;
  sesgo sistemático posible de SAM 2 (bordes suaves/fragmentación) listado
  como amenaza de constructo (EDA A3). Áreas de lesión comparables entre
  train+val y test (KS p = 0,382) ⇒ misma población.

### Versión, hashes y trazabilidad

| Artefacto | Versión / fecha | SHA-256 |
|---|---|---|
| `split_manifest_v2.json` | v2, sellado **2026-10-20T20:01:13** | ver archivo (`hash_integridad_v2`) |
| `train.csv` | v2 | `8d88e56a5ff6dd2afe0b1b8c599c847b7762c4cfd948e7c438eae567adc1b6a5` |
| `val.csv` | v2 | `a63050d3c33aa5d3cdc5efadbd807f25ed5631a123c09a6224c01370b4f6264d` |
| `test_pos.csv` | v2 | `8032fcbae326032987f3f623f6f9e039f6645913b0ebc3de9a60c1401f23f8a6` |
| `test_neg.csv` | v2 | `c3120ba4abc99c313880f4832094201022f77d94816d19d23f784d9c19d51d0d` |
| `test_pos_reserva.csv` | v2 | `a2e62c71cf89ce2869b6fe9e12f9fc97da55d938a0b8cc48b4b2bbd79fc7e348` |
| `pseudo_gt_manifest_v2.json` | v2, 2026-10-20 | `c3fe381c22ea3d7790a029fe4a485b27…` (y `script_sha256`/`checkpoint_sha256` internos) |
| `ANOTACION_METRICAS.json` (GT test) | validado 2026-10-20T23:42:07 | `ed872f341b5c3f9f075328d3a0a07b12…` |

Verificación automática en cada ejecución del EDA (Sección 2, A8): los CSV se
comparan por SHA-256 contra `hash_integridad_v2` y la membresía la define el
CSV — si un byte cambia, el EDA falla. **La lógica es de solo lectura:
los CSV sellados jamás se modifican** (remapeo únicamente en memoria).

### Referencias del dataset

```bibtex
@inproceedings{liu2020rethinking,
  title     = {Rethinking Computer-Aided Tuberculosis Diagnosis},
  author    = {Liu, Yun and Wu, Yu-Huan and Ban, Yunfeng and Wang, Huifang and Cheng, Ming-Ming},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {2646--2655},
  year      = {2020},
  doi       = {10.1109/CVPR42600.2020.01270}
}
```
