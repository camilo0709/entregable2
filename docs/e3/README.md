# Entregable 3 — LACCA-UNet: material de reproducibilidad

**Autores:** Mujica Camilo y Romero Carlos.

Este directorio agrupa, sin tocar el código del Entregable 2, todo lo
generado en el Entregable 3 (segmentación de lesiones de TB en TBX11K).

| Ruta | Contenido |
|---|---|
| `codigo/e3/` | Modelo original (`original_model.py`), registro de modelos con las ablaciones, búsqueda de hiperparámetros, driver, evaluación del test sellado, estadística (Holm, Cliff δ), ONNX, análisis extra y validación externa |
| `codigo/e3/notebooks/` | Notebooks de Colab 01–05 (sin salidas), en el orden de ejecución |
| `codigo/e3/tests/` | Pruebas unitarias del modelo original |
| `docs/e3/` | Protocolo congelado (`protocolo.md`) y enmiendas de preregistro |
| `resultados/e3/` | Tablas y figuras pequeñas: HPO, benchmark, comparaciones, calibración, robustez, Grad-CAM |

Los pesos (`.pt`, `.onnx`) y los datos no se versionan.

## Resultados principales (test sellado, 80 positivas, una sola lectura)

- Benchmark de 9 modelos × 5 semillas: attention_unet (U-Net + scSE) y
  unet_pp lideran; el resto queda entre 0,47 y 0,51 de Dice.
- LACCA-UNet (10 semillas, Dice 0,565) empata con attention_unet y unet_pp
  (Δ de 0,006 y 0,008; p Holm = 1) y supera a mednext (Δ = 0,055;
  p Holm < 0,001).
- Las tres ablaciones (eje fijo, sin asimetría, atención global) no son
  distinguibles del modelo completo: no se demuestra aporte del eje
  aprendido ni del residuo de asimetría.
- Calibración y robustez: solo semilla 42, análisis exploratorio.

## Limitaciones declaradas

nnU-Net queda fuera de la evaluación (entrenamiento incompleto);
`lacca_global` se entrenó con batch 4; el HPO mezcla paciencia 25 y 15
(enmienda E7); el top 3 se eligió con las mismas semillas que luego se
ampliaron (sesgo de selección pequeño).

## Nota de integración

`codigo/e3/` es autocontenido y **no reemplaza** los módulos de `codigo/`.
Varios archivos con el mismo nombre difieren entre ambas ramas de código
(p. ej. `modelos_tb.py`, `metricas_tb.py`, `driver_entrenamiento.py`);
en particular `DELTA_DICE` es 0,02 en `codigo/metricas_tb.py` y 0,01 en
`codigo/e3/metricas_tb.py`. Conviene unificarlos antes de la entrega.
