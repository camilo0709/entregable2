# Artefactos de inferencia

Los 14 modelos exportados a ONNX (opset 17, entrada `[1,1,512,512]` con z-score por imagen, salida en logits) pesan
entre 25 MB y 351 MB, más de lo que admite un commit de GitHub. Se publican como *assets* del Release
`pesos-onnx-v1`; el tablero los descarga bajo demanda y verifica su SHA-256 contra `perfil_modelos.json`.

- `perfil_modelos.json`: parámetros, GFLOPs, latencia en CPU, SHA-256 del checkpoint y del ONNX, y resultado de la
  verificación de ida y vuelta (tolerancia 1e-4).
- Variable `TB_WEIGHTS_DIR`: carpeta local con los `.onnx` (omite la descarga).
- Variable `TB_WEIGHTS_URL`: URL base de los pesos.
- UNETR (88 M parámetros) no se sirve en el plan gratuito de Render por memoria; sus resultados están precalculados.
