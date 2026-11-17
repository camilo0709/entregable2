# Sección 9 · Trazabilidad de correcciones (Entregable 3)

Esta sección responde, punto por punto, a la retroalimentación del Entregable 2
(nota 3,5) y funciona como **Bloque A** del Entregable 3. Cada fila declara la
observación recibida, la acción correctiva, dónde queda la evidencia y el
estado real al cierre de esta edición. Los estados no se maquillan: lo que
todavía depende de la fase de test o de acceso institucional se marca como
**En curso** o **Pendiente**, no como resuelto.

**Leyenda:** **Resuelto** = cerrado y verificable en el repositorio;
**En curso** = código o estructura listos, ejecución completa pendiente;
**Pendiente** = bloqueado por un recurso externo.

| # | Observación del Entregable 2 | Acción correctiva | Evidencia y ubicación | Estado |
|---:|---|---|---|---|
| 1 | nnU-Net no aportó resultados y Attention U-Net salió duplicado de U-Net (transparencia valorada, pero hueco en la comparación) | nnU-Net se declara **no disponible** (0/5) y no se imputa. El duplicado de Attention se diagnostica: `smp.Unet` ignoraba `attention_type`; se corrige a `decoder_attention_type="scse"` (24,54 M parámetros, 31,03 GMACs) y se invalidan sus 5 corridas. Se reentrena en la fase final | Sección 4.1.2 (contabilidad), 4.2.1 (registro), 6.1 y 6.7 (limitaciones); `codigo/modelos_tb.py`; `resultados/attention_unet/` y `resultados/nnunet/`; **E3:** nnU-Net se mantiene no disponible (limitación declarada); Attention U-Net corregido reentrenado con 10 semillas: Dice test 0,5595 (`resultados/e3/resumen_test.csv`) | Resuelto |
| 2 | La fase de test sellado no se evaluó | `evaluar_test.py` infiere `test_pos` una sola vez, con umbral fijo 0,5 y el checkpoint de la mejor época; escribe `por_imagen_test.csv`, `dice_test` y, opcionalmente, el control OOD en `test_neg`. Validado con DeepLabV3+ s42: 80 casos, Dice 0,54001, 216 s en CPU | Sección 4.1.2 (garantías del test), 6 (bloque de test); `codigo/evaluar_test.py`; **E3:** Test sellado evaluado en 13 modelos/variantes (80 positivas + 40 negativas): `resultados/e3/resumen_test.csv`, capítulo e3_03 | Resuelto |
| 3 | Faltó la calibración de probabilidades | `calibracion.py` calcula Brier y ECE (15 bins) global, en lesión y en fondo, y ajusta **temperature scaling solo en validación**. No cambia el umbral 0,5 del preregistro. Prueba con DeepLabV3+ s42: T = 2,04; Brier global 0,0573 → 0,0535 en 3 imágenes de val | Sección 7, Tabla B (fila de calibración); `codigo/calibracion.py`; **E3:** Calibración (ECE, temperature scaling) de 5 modelos, semilla 42: `resultados/e3/extra/*/calibracion.json` | Resuelto |
| 4 | Faltó la robustez ante perturbaciones | `robustez.py` aplica 10 condiciones antes del z-score: ruido gaussiano σ ∈ {5, 10, 20}, desenfoque σ ∈ {1, 2}, gamma ∈ {0,8; 1,25}, rotación ±10° y limpio. Reporta Dice, IoU, HD95 y ASSD por condición. Prueba con 2 imágenes: ruido σ20 cae ΔDice −0,253; rotación +10° da +0,032 | Sección 7 (evaluación ampliada); `codigo/robustez.py`; **E3:** Robustez de 5 modelos, semilla 42: `resultados/e3/extra/*/robustez.csv` (una semilla; indicativo) | Resuelto |
| 5 | Faltó Grad-CAM y su validación espacial | `gradcam.py` engancha el último mapa del encoder, calcula el score dentro del GT y mide masa del CAM dentro del GT frente al área del GT, precisión del top-20 % y solape con la caja TBX11K. Prueba en `tb0018`: masa 0,136 vs área 0,050 (lift +0,086), top-20 % 0,184 | Sección 7, Tabla B (fila de Grad-CAM); `codigo/gradcam.py`; **E3:** Grad-CAM de 5 modelos: `resultados/e3/extra/*/gradcam.png`, `confiabilidad.csv` | Resuelto |
| 6 | Faltó el diagnóstico de sobreajuste | `sobreajuste.py` mide la pérdida DiceBCE del checkpoint en train y val (margen δ_g = 0,05), el optimismo de selección de época (mejor val − media de la curva, z del pico) y la brecha val–test cuando exista `dice_test`. Prueba con DeepLabV3+ s42 (8 imágenes/split): pérdida 0,398 vs 0,461 (brecha +0,063, moderada); Dice 0,496 vs 0,475; optimismo +0,081 (z = 1,30 < 2,22 esperado) | Sección 7, Tabla B (filas de brecha y optimismo); `codigo/sobreajuste.py`; **E3:** Brecha val–test visible en `resultados/e3/resumen_test.csv` (dice_val vs dice_test) | Resuelto |
| 7 | El PRISMA no tenía los conteos de Scopus e IEEE | No se inventan cifras: Scopus e IEEE quedaron como `s/d` en el Entregable 2 por depender del acceso institucional. La estructura, las ecuaciones de conteo y las fuentes sí están completas | Sección 3 (PRISMA); `codigo/build_figuras_rev.py` (`COUNTS`); **E3:** conteos ejecutados el 2026-11-16 (Scopus 656/2 577/580; IEEE 53/584; arXiv 12/1) y búsqueda de novedad en e3_01. Los números de cribado posteriores (duplicados, exclusiones, incluidos) no se recalcularon | Resuelto con limitación |
| 8 | Faltaban los resultados crudos en el repositorio | Se incorporan al repositorio `resultado.json`, `por_imagen.csv`, checkpoints y `resumen_runs.csv`; el traslado desde Colab se hace con el zip liviano `resultados_stats.zip` (no con la carpeta Drive completa, que sale incompleta). En local hay 21 `resultado.json` y 16 checkpoints; falta el zip completo | `resultados/`; Sección 8 (reproducibilidad); `codigo/setup_colab.ipynb`; **E3:** Resultados crudos incorporados en `resultados/e3/` (87 corridas, `experiments/tabla_maestra.csv`); checkpoints y ONNX pesados no se versionan (SHA-256 en `artifacts/perfil_modelos.json`) | Resuelto |
| 9 | Las cifras no cuadraban: 13 modelos y 55 corridas frente a 10 modelos y 50 corridas | Se fija una **tabla canónica** en la Sección 4.1.2: 13 entradas del registro (10 arquitecturas + 3 baselines), 11 entrenables, 55 planificadas, 50 completadas (45 de arquitecturas + 5 de `unet_fuerte`), 40 válidas para RQ1 (8 × 5). Se alinean las Secciones 3, 6, 7 y el anexo | Sección 4.1.2 (tabla), 6.1 (diseño ejecutado), 7 (Paso 7), anexo B9 y C8; **E3:** Cifras recalculadas: 14 modelos/variantes, 90 corridas completadas (`resultados/e3/auditoria_resultados.md`) | Resuelto |

## 9.1 Cómo verificar cada corrección

Los cuatro módulos nuevos se prueban con un checkpoint real y un subconjunto
pequeño, de modo que la comprobación no depende de Colab:

```bash
python codigo/calibracion.py --modelo deeplabv3p --semilla 42 --limite 8
python codigo/robustez.py  --modelo deeplabv3p --semilla 42 --limite 4
python codigo/gradcam.py   --modelo deeplabv3p --semilla 42 --indices 0
python codigo/sobreajuste.py --modelo deeplabv3p --semilla 42 --limite 8
```

La contabilidad se verifica contra el registro, no contra el texto:

```bash
python codigo/estadistica_tb.py --resultados resultados
```

El script detecta los modelos con corridas válidas y escribe la matriz
modelo × semilla; si la matriz no tiene 8 filas y 5 columnas, la tabla de la
Sección 4.1.2 y las cifras de la Sección 6 no están alineadas.

## 9.2 Estado de cierre del Bloque A

Cerrado con evidencia en `resultados/e3/`: puntos 1–6, 8 y 9. Quedan **abiertos**:

1. **Punto 7:** conteos de Scopus e IEEE, que dependen del acceso institucional de los autores.
2. **nnU-Net** (parte del punto 1): sigue sin evaluarse y queda como limitación declarada.
3. **Alcance:** calibración, robustez y Grad-CAM se hicieron con una sola semilla y 5 modelos (no los 8 originales); es evidencia indicativa.

Las cifras del Entregable 2 (validación) fueron sustituidas por las de test sellado y por los análisis con 10 semillas descritos en los capítulos e3_00 a e3_05. El resultado principal es negativo en superioridad (R3 no cumplido).
