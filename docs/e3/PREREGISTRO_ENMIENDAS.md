# ENMIENDAS AL PREREGISTRO — ENTREGABLE 2

> **Regla de oro** (`docs/PREREGISTRO_ENTREGABLE2.md`): cualquier desviación se
> registra aquí con fecha, motivo y **antes** de observar resultados del test
> que la motiven. Ninguna enmienda de este archivo se adoptó después de ver
> métricas de test.

---

## E1 — `attention_unet` no estaba aplicando atención (corregido)

- **Fecha:** 2026-11-13
- **Estado:** ✅ corregida (re-entrenamiento de las 5 semillas en curso)
- **Motivo:** `codigo/modelos_tb.py` construía `smp.Unet` con
  `attention_type="scse"`, parámetro **legado que segmentation-models-pytorch
  >= 0.3 ignora**. Consecuencia: `attention_unet` producía predicciones
  **byte a byte idénticas** a `unet` (mismo SHA-256 de `por_imagen.csv` en las
  5 semillas), por lo que el benchmark habría reportado dos filas del mismo
  modelo.
- **Cambio:** se sustituye por `decoder_attention_type="scse"` (descrito
  aquí originalmente como "gates de Oktay"; **corrección en E6**: es scSE).
- **Alcance:** se re-entrenan las 5 semillas de `attention_unet` con el
  protocolo intacto (mismos datos, pérdida, épocas, semillas y umbral). El
  resto de modelos no se toca.
- **Transparencia:** el test **no se había evaluado** cuando se detectó y
  corrigió (la carpeta del benchmark solo tenía métricas de validación). Las
  métricas de `attention_unet` anteriores quedan archivadas como
  `estado="duplicado_de_unet"` y no entran al análisis.

## E2 — El GT del test es plata (D1-R1), no anotación manual

- **Fecha de la decisión original:** 2026-10-20 (D1-R1, aprobada)
- **Estado:** ✅ vigente
- **Motivo:** el cuerpo del preregistro (§3) sigue describiendo el test como
  "GT manual"; la revisión D1-R1 aprobada antes de entrenar sustituyó ese GT
  por la **pseudomáscara SAM 2 congelada** (silver standard declarado).
- **Precisión:** el GT vigente es `datos/ground_truth_test/<id>.png`
  (80 imágenes), generado por `codigo/generar_pseudo_gt.py` con el manifiesto
  `datos/pseudo_gt_manifest_v2.json`. La validación humana de la submuestra de
  20 imágenes dio **Dice 0.9879** (mín. 0.7583), registrado en
  `datos/ground_truth_test/ANOTACION_METRICAS.json`, y se reporta como
  **techo de desempeño**.
- **Consecuencia declarada:** las métricas del benchmark miden **imitación del
  pipeline teacher** (SAM 2 + caja de TBX11K), no calidad clínica absoluta.
  Debe constar en el Data Card y en el libro.

## E3 — nnU-Net como pipeline propio (5 folds ↔ 5 semillas)

- **Fecha:** 2026-10-21 (implementación), formalizada 2026-11-13
- **Estado:** ✅ declarada; ejecución pendiente
- **Equivalencias:** fold 0→semilla 42, 1→123, 2→777, 3→2024, 4→31415;
  configuración 2d; trainer de 50 épocas (`nnUNetTrainer_50epochs` o trainer
  propio equivalente); pool train+val = 194 npz con GT plata.
- **Test:** 80 positivas (GT plata) + 40 negativas (máscara vacía), evaluadas
  por fold. El score de detección es la masa de probabilidad del canal lesión,
  análoga al resto.
- **Limitación declarada:** las 5 "semillas" de nnU-Net son 5 particiones
  internas distintas, no 5 inicializaciones sobre el mismo split. Se reporta
  como diferencia de diseño, no como error.
- **Detalle operativo:** ver `docs/NNUNET_PIPELINE.md`.

## E4 — `unet_fuerte` es hoy un duplicado de `unet`

- **Fecha:** 2026-11-13
- **Estado:** ⚠️ declarado; no entra al Friedman (ya estaba excluido)
- **Hallazgo:** el baseline "U-Net fuerte" se entrenó con la **misma
  configuración** que `unet` (mismos hiperparámetros en `resultado.json`), por
  lo que sus 5 corridas son idénticas. El preregistro lo describe como "U-Net
  con búsqueda de hiperparámetros", que no llegó a ejecutarse.
- **Decisión:** no se presenta como evidencia de tuning. En Tabla A se
  etiqueta como duplicado de `unet`; si se quiere una cota superior CNN real,
  debe hacerse la búsqueda y registrarse como nueva corrida (no modifica el
  ranking de los 10 modelos).

## E5 — Orden de evaluación

- **Fecha:** 2026-11-13
- **Estado:** ✅ cumplido
- El GT plata estaba congelado y el preregistro sellado antes de esta sesión.
  La evaluación de test que produce este repositorio es la **primera** lectura
  del test por parte de los modelos; ningún modelo se re-entrena mirando esas
  métricas (la selección de época sigue siendo solo `dice_val`).

## E6 — `attention_unet` pasa a llamarse "U-Net + scSE" (no son gates de Oktay)

- **Fecha:** 2026-11-13
- **Estado:** ✅ aplicada en `codigo/modelos_tb.py`
- **Motivo:** E1 describió `decoder_attention_type="scse"` como "gates de
  Oktay". Es incorrecto: scSE es *Concurrent Spatial and Channel
  Squeeze-and-Excitation* (Roy et al., MICCAI 2018), un módulo de
  recalibración del decoder distinto de las compuertas de atención aditiva de
  la Attention U-Net (Oktay et al., 2018).
- **Cambio:** el nombre de presentación pasa de "Attention U-Net" a
  "U-Net + scSE" y la referencia a Roy et al. 2018. La clave interna
  `attention_unet` **no cambia**, para no romper `resumen_runs.csv`,
  `analisis_estadistico.py` ni `evaluar_test.py`. La arquitectura y el
  protocolo de entrenamiento no cambian respecto a E1.
- **Alcance:** solo etiquetado (tablas, figuras, libro, y la fila 2 del
  benchmark en `PREREGISTRO_ENTREGABLE2.md`, que se lee ahora como
  "U-Net + scSE"). No se afirma que el benchmark incluya la Attention U-Net
  de Oktay.
- **Efecto sobre resultados:** ninguno. Los resultados de `attention_unet`
  anteriores a E1 siguen archivados como duplicado de `unet`
  (`resumen_runs.csv` aún los contiene; `limpiar_resultados.py` los detecta).
- **Transparencia:** adoptada antes de cualquier lectura del test por parte
  del modelo corregido; el re-entrenamiento de E1 sigue pendiente.

## E7 — Presupuesto reducido: 6 intentos de HPO, paciencia 15 y `drop_last`

- **Fecha:** 2026-11-14
- **Estado:** ✅ aplicada en `hpo_busqueda.py` y `driver_entrenamiento.py`;
  `docs/protocolo.md` actualizado antes de congelarse.
- **Motivo:** reducir tiempo de cómputo (Colab). Cambia la búsqueda de 8 a 6
  intentos por modelo (intento 0 por defecto + 5 aleatorios; los intentos
  0–5 son idénticos a los de la versión de 8, pues el muestreo es determinista
  por modelo e intento) y la paciencia del early stopping de 25 a 15 épocas,
  para HPO y entrenamiento final por igual.
- **Corrección técnica:** el DataLoader de entrenamiento usa `drop_last=True`.
  Con lote 4 el último lote de 165 imágenes tenía 1 muestra y DeepLabV3+
  falla en BatchNorm (1×256×1×1). Descarta como máximo `batch-1` imágenes por
  época (barajadas distintas cada época).
- **Intentos ya hechos antes de E7** (paciencia 25, sin `drop_last`) se
  conservan si su índice es < 6; se declara que entrenaron con paciencia 25.
- **Efecto sobre hipótesis y pruebas:** ninguno; no se ha leído el test.
