# PROTOCOLO DEL ENTREGABLE 3 (congelado antes de los experimentos finales)

> Este archivo se congela antes de entrenar las semillas finales y de leer el
> test. `colab_02_entrenamiento_final.ipynb` guarda su SHA-256 en
> `resultados_e3/protocolo.sha256`; los notebooks 03 y 04 abortan si cambia.
> Cualquier desviación se registra como enmienda nueva
> (`docs/PREREGISTRO_ENMIENDAS.md`, E7 en adelante), con fecha y antes de ver
> métricas de test.

## 1. Pregunta e hipótesis
- Modelo original: **LACCA-UNet** (`codigo/original_model.py`): U-Net resnet34 con
  atención cruzada contralateral de eje aprendido en las etapas 3 y 4 del encoder.
- **H1:** el modelo original obtiene mayor Dice en el test sellado (80 positivas,
  GT plata SAM 2) que cada uno de los 3 mejores modelos del benchmark.
- **H0:** Dice medio del original − Dice medio del rival ≤ 0.
- Margen de relevancia: diferencia de medias > 0.01 de Dice.

## 2. Datos
- Split sellado 165 train / 29 val / 80 test_pos / 40 test_neg / 64 reserva
  (`datos/splits`, manifiesto v2). Train/val con pseudomáscara SAM 2; test con GT
  plata (D1-R1; techo validado por humanos 0.9879 en 20 imágenes).
- Validación externa en Montgomery, Shenzhen y JSRT (máscaras de **pulmón**, no de
  lesión): reparto fijo 70/15/15 (semilla 2026); el 15 % es test retenido.
  Mide transferencia a otra anatomía/fuente con el mismo protocolo, no
  segmentación de lesiones.

## 3. Entrenamiento (idéntico para todos)
- Pérdida DiceBCE 0.5/0.5; AdamW; coseno; ≤150 épocas; paciencia 15 (E7); umbral 0.5;
  selección de época solo por `dice_val`; encoder sin preentrenar.
- Semillas base: 42, 123, 777, 2024, 31415 (todos los modelos).
- Semillas extra: 7, 11, 99, 1234, 9999, solo para el original y el top 3 → **10 semillas**.

## 4. Búsqueda de hiperparámetros
- 6 intentos por modelo (E7: intento 0 = configuración por defecto + 5 aleatorios),
  1 semilla (42) por intento, selección solo por `dice_val`. Mismo presupuesto
  para los 10 modelos buscados: `unet_fuerte`, U-Net + scSE, U-Net++, DeepLabV3+,
  TransUNet*, Swin-UNet*, UNETR, SegFormer, MedNeXt* y el original.
- `unet` se entrena sin búsqueda; `unet_fuerte` es la misma red con búsqueda.
- nnU-Net usa su propio pipeline (E3) sin búsqueda de este tipo.

## 5. Top 3
Los 3 mejores por media de `dice_val` en las 5 semillas base (validación),
entre los modelos del benchmark entrenables en el driver (sin nnU-Net).
Regla ejecutable: `codigo/elegir_top3.py`.

## 6. Evaluación
- Primaria: Dice medio por imagen en test_pos (media sobre semillas).
- Secundarias: IoU, HD95, ASSD, sensibilidad/especificidad (test_pos + test_neg),
  AUC de detección (masa de probabilidad).
- El test se lee **una vez** por checkpoint congelado (`evaluar_test.py`).

## 7. Pruebas estadísticas (`codigo/estadistica_e3.py`)
Original vs cada rival del top 3: T1 Wilcoxon pareado por imagen, T2 bootstrap
pareado (10 000 remuestreos, IC 95 %), T3 Mann-Whitney sobre medias por semilla;
Friedman omnibus. **Holm** sobre los 9 p-valores, α = 0.05. Efecto: δ de Cliff,
relevante si |δ| ≥ 0.147. "Gana" = significativo tras Holm, |δ| ≥ 0.147 y
diferencia a favor.

## 8. Ablaciones (5 semillas c/u, configuración del modelo completo)
`lacca_eje_fijo` (sin eje aprendido), `lacca_sin_asimetria` (D = A),
`lacca_global` (atención global). Misma estadística que la sección 7.

## 9. Calibración, robustez e interpretabilidad (`codigo/analisis_extra.py`)
ECE de píxel (15 bins) antes y después de escalado por temperatura (T ajustada
solo en validación); robustez a ruido, desenfoque, contraste y brillo;
Grad-CAM. Se reporta para original, top 3 y U-Net (semilla 42).

## 10. Despliegue (`codigo/exportar_onnx.py`)
ONNX opset 17, 1×1×512×512; ida y vuelta con tolerancia 1e-4 en probabilidad;
`metadata.json`; parámetros, FLOPs, latencia (CPU ORT y GPU) y memoria.

## 11. Qué no se hace
No se ajusta el umbral ni la época con el test; no se repiten corridas descartando
semillas; no se cambian hiperparámetros tras ver el test; toda exclusión o
re-entrenamiento se registra como enmienda.
