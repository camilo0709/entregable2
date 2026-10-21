# PROTOCOLO DE ANOTACIÓN MANUAL — TEST SET (D1)

- **Derivado de:** `DECISION_D1_GROUND_TRUTH.md`
- **Objetivo:** producir el ground truth píxel-a-píxel de las **80 imágenes**
  de `datos/splits/test_pos.csv`
- **Estado:** ⬜ pendiente de ejecución
- **Precondición:** el split está **sellado**; no añadir/quitar imágenes del test

---

## 1. Alcance

| Item | Valor |
|---|---|
| Imágenes positivas a anotar | **80** (`test_pos.csv`) |
| Reserva disponible | 64 (`test_pos_reserva.csv`) |
| Negativos (40, `test_neg.csv`) | **no requieren anotación** — máscara vacía por construcción |
| Resolución | 512 × 512 px (original TBX11K, sin reescalado) |
| Formato de salida | PNG binario 0/255, mismo `id` que la imagen |
| Directorio de salida | `datos/ground_truth_test/` |

---

## 2. Qué anotar (definición de la lesión)

**Objeto:** región pulmonar afectada por tuberculosis visible en la radiografía.

Reglas:

1. **Incluir** infiltrados, cavidades, nódulos, atelectasias y opacaciones
   compatibles con TB dentro de los campos pulmonares.
2. **Excluir** estructuras no patológicas: costillas, clavículas, mediastino,
   corazón, diafragma, bordes extrapulmonares.
3. Si hay **múltiples focos** separados → una máscara binaria que une todos los
   focos (la segmentación es binaria; el número de focos se guarda aparte).
4. Si la lesión es **difusa/bilateral** → anotar la extensión que el anotador
   juzgue afectada, aplicando el criterio de "si un radiólogo señalaría esta
   zona como afectada".
5. Bordes: **suavemente difusos son aceptables**; el criterio es reproducibilidad
   perjudicada por sobre-ajuste fino, no precisión submilimétrica.

---

## 3. Herramienta

Usar un visor que permita dibujar/poligonizar sobre la imagen. Opciones
recomendadas (cualquiera sirve, siempre exportando PNG binario 512×512):

- **ITK-SNAP** (gratis, estándar en segmentación médica) — recomendado
- **3D Slicer** (gratis)
- **CVAT** o **Label Studio** si se monta un flujo web
- Cualquier editor con selección por pincel + umbral, **siempre** verificando
  que el PNG final sea exactamente 512×512 y binario

### 3.1 Exportar las imágenes de trabajo

```powershell
python codigo/exportar_imagenes_para_anotar.py
```

(Este script copia las 80 PNG de `test_pos.csv` a
`datos/anotacion_workspace/imagenes/` para no tocar el dataset original.)

---

## 4. Procedimiento paso a paso

| Paso | Acción | Control de calidad |
|---|---|---|
| 1 | Anotador A segmenta las 80 imágenes | — |
| 2 | **Doble anotación** en submuestra aleatoria de **20 imágenes** (25 %) por anotador B | calcular Dice inter-anotador |
| 3 | Revisar pares con Dice inter-anotador **< 0,60** → consenso | resolver discrepancias |
| 4 | Validar formato de cada PNG exportado | script de validación (ver §6) |
| 5 | Registrar métricas de calidad en el manifiesto | congelar GT |

### 4.1 Criterio de aceptación de calidad

| Métrica | Umbral |
|---|---|
| Dice inter-anotador (media en submuestra) | **≥ 0,70** |
| Imágenes con máscara vacía por error | **0** (todo test_pos es positivo) |
| PNG con tamaño ≠ 512×512 | **0** |
| PNG con valores distintos de {0, 255} | **0** |
| Máscara con < 100 píxeles positivos | marcar para revisión |

Si Dice inter-anotador < 0,70 → reescribir las reglas de §2 y reanotar la
submuestra afectada **antes** de congelar el GT.

---

## 5. Reproducibilidad y cegamiento

1. **Cegamiento:** el anotador **no debe ver** las predicciones de ningún modelo
   ni las pseudomáscaras SAM 2 durante la anotación.
2. **Orden aleatorio:** anotar en orden aleatorio fijado con semilla 42
   (ya fijado por `generar_split_entregable2.py`).
3. **Inmutabilidad:** una vez validado, el GT se congela. Cualquier cambio
   posterior debe registrarse con fecha y motivo en
   `ground_truth_test/ANOTACION_CAMBIOS.md`.
4. **Sellado:** `split_manifest.json` queda `"sealed": true`; no reasignar
   imágenes entre train/val/test después de este punto.

---

## 6. Validación (script)

```powershell
python codigo/validar_ground_truth_test.py
```

Comprobaciones que debe ejecutar:

- [ ] Existen exactamente 80 PNG en `datos/ground_truth_test/`
- [ ] Cada PNG mide 512×512
- [ ] Valores ⊆ {0, 255}
- [ ] Ninguna máscara vacía
- [ ] Todos los `id` de `test_pos.csv` están cubiertos (sin huecos)
- [ ] Áreas en rango razonable (no < 100 px ni > 40 % de la imagen sin revisar)
- [ ] Dice inter-anotador ≥ 0,70 en la submuestra de 20

---

## 7. Estructura de salida esperada

```
datos/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   ├── test_pos.csv            ← sellado, 80 ids
│   ├── test_neg.csv            ← sellado, 40 ids
│   ├── test_pos_reserva.csv
│   └── split_manifest.json     ← "sealed": true
├── ground_truth_test/
│   ├── tb0018.png              ← máscara binaria 512x512
│   ├── tb0030.png
│   ├── ...                     (80 archivos)
│   ├── ANOTACION_CAMBIOS.md
│   └── ANOTACION_METRICAS.json ← Dice inter-anotador, conteos
└── anotacion_workspace/
    └── imagenes/               ← copias para anotar
```

---

## 8. Esfuerzo estimado

| Fase | Tiempo |
|---|---|
| Preparar exportación + herramienta | 0,5 h |
| Anotación A (80 img × 4 min media) | **5,3 h** |
| Doble anotación B (20 img × 4 min) | 1,3 h |
| Revisión de discrepancias + consenso | 1 h |
| Validación y congelado | 0,5 h |
| **Total** | **≈ 8,5 h** |

> Si el tiempo es limitante, se puede reducir a **60 imágenes** manteniendo
> potencia para Friedman (k=10 tratamientos); actualizar entonces
> `N_TEST_POS` en el generador y **regenerar el manifiesto declarándolo**.

---

## 9. AJUSTE R1 (2026-10-20) — VALIDACIÓN HUMANA DE SUBMUESTRA (no anotación completa)

> Aplica a la revisión **D1-R1** (`DECISION_D1_GROUND_TRUTH.md` §R1): el GT
> completo del test_pos (80) lo genera el pipeline congelado SAM 2
> (`codigo/generar_pseudo_gt.py`). La anotación manual de las 80 se **sustituye**
> por la **validación humana de una submuestra de 20**, que funciona como
> **techo de desempeño** del benchmark.

### 9.1 Qué se hace ahora (≈1–1,5 h en vez de ≈8,5 h)

| Paso | Acción | Control de calidad |
|---|---|---|
| 1 | Generar el GT plata de las 80 con `generar_pseudo_gt.py` | manifiesto + hashes |
| 2 | `validar_ground_truth_test.py` (formato/áreas) | checklist §6 |
| 3 | Seleccionar submuestra estratificada de **20** (`preparar_submuestra_validacion.py`, semilla 42, estratificada por `n_boxes`) | CSV de submuestra |
| 4 | El usuario **corrige o aprueba** cada máscara SAM 2 precargada (visor con pincel; ver §9.3) | máscara humana de las 20 |
| 5 | `validar_ground_truth_test.py --anotador-b datos/submuestra_validacion/mascaras_humano` → **Dice(humano, SAM 2)** | ≥ 0,70 media |
| 6 | Registrar el Dice como **techo de desempeño** en el manifiesto y el book | congelado |

### 9.2 Criterios de aceptación (se mantienen umbrales)

| Métrica | Umbral | Acción si falla |
|---|---|---|
| Dice medio (humano vs SAM 2) en la submuestra | **≥ 0,70** | reforzar post-proceso del pipeline y regenerar; si persiste, escalar a revisión con radiólogo |
| Máscaras humanas vacías por error | 0 | rehacer la imagen |
| PNG humano 512×512 binario | obligatorio | script de validación rechaza |

### 9.3 Cambios declarados respecto al protocolo ciego original

1. **El cegamiento §5.1 ya no aplica a la submuestra:** el validador ve la
   máscara SAM 2 y la corrige (esto es validación-por-corrección, no
   anotación independiente). Se declara en el informe: el Dice resultante es
   una **cota conservadora** del acuerdo (ver la máscara y corregirla suele
   producir mayor acuerdo que anotar a ciegas), por lo que usarlo como techo
   de desempeño es la interpretación prudente.
2. El validador **no es radiólogo**; su rol es de consistencia interna
   (¿la región señalada por el anotador original de TBX11K se ve razonable?),
   no de veracidad clínica. Esto se explicita en el Data Card.
3. La submuestra se elige **antes** de ver resultados de cualquier modelo del
   benchmark (semilla 42, estratificada por `n_boxes`) → sin selección
   favorable.

### 9.4 Exportación

```bash
python codigo/preparar_submuestra_validacion.py
```

Genera `datos/submuestra_validacion/` con `imagenes/`, `mascaras_sam2/`,
`overlays/` (imagen + máscara translúcida + contorno de la caja original) y
`submuestra.csv`. El validador exporta sus correcciones como PNG binarios
0/255 en `mascaras_humano/<id>.png`.

### 9.5 Resultados de la validación (2026-10-20)

Ejecución real de los pasos 1–6 de §9.1 por el validador
(Camilo Andrés Mujica Escorcia, revisando los 20 overlays uno a uno):

| Elemento | Resultado |
|---|---|
| Cobertura | 20/20 máscaras humanas entregadas |
| Aprobadas tal cual (copia idéntica de SAM 2) | 19/20 |
| Corregidas / variantes elegidas | 1/20 → `tb0393` |
| **Dice(humano, SAM 2) — media** | **0,9879** (std 0,0527) |
| **Mínimo** | **0,7583** (`tb0393`) |
| Umbral §9.2 | ≥ 0,70 → **CUMPLE** |

El único defecto detectado: `tb0393` traía la caja derecha **fragmentada**
(SAM 2 eligió por empate de score 0,219 vs 0,215 la candidata con dice-vs-caja
0,476 en vez de la sólida con 0,718). El validador eligió entre 4 variantes
generadas la `v2` (ambas cajas sólidas). El GT plata **congelado NO se
modifica** — la variante solo vive como máscara humana, por lo que el 0,7583
de esta imagen penaliza con honestidad la calidad del plata en ese caso
(el registro completo está en `ground_truth_test/ANOTACION_METRICAS.json`).

**Techo de desempeño del benchmark: Dice medio 0,99 sobre test_pos.**
Interpretación declarada (§9.3.1): validación-por-corrección ⇒ cota
conservadora; se declara en Data Card y book.
