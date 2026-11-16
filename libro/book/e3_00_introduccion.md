# Entregable 3 · LACCA-UNet: atención cruzada contralateral con eje aprendido

**Autores:** Mujica Camilo y Romero Carlos.

## Resumen

Proponemos **LACCA** (*Learned-Axis Contralateral Cross-Attention*), un módulo que compara cada región del
mapa de características con su reflejo respecto a un eje de simetría torácico **aprendido**, y lo integra
en un U-Net con encoder ResNet34 (**LACCA-UNet**, etapas 3 y 4). Se evalúa en TBX11K (segmentación de
lesiones de TB, GT de plata SAM2) con 5 y 10 semillas, HPO, test sellado, ablaciones, pruebas de Holm,
calibración, robustez y validación externa.

**Veredicto honesto.** LACCA-UNet es *competitivo*, no superior: Dice de test 0.5651 ± 0.022 (10 semillas)
frente a 0.5595 (Attention U-Net) y 0.5574 (U-Net++), diferencias no significativas (p Holm = 1). Solo supera
de forma significativa a MedNeXt (+0.0546). Por tanto se cumplen R1 y R2 pero **no R3** (superioridad
estadística sobre el mejor baseline); según el Cuadro 7 del enunciado la nota queda limitada a **4.0**.

## Preguntas e hipótesis

- **H1.** Incorporar el reflejo contralateral con eje aprendido mejora el Dice de lesión frente al U-Net base.
- **H2.** El eje aprendido y el residuo de asimetría aportan frente a variantes fijas (ablación).
- **H3.** LACCA supera al mejor baseline del benchmark con significancia (Holm, Cliff δ ≥ 0.147, margen 0.01).

H1 se sostiene (frente a U-Net: 0.5651 vs 0.4985). H2 y H3 **no se sostienen** con los datos disponibles.

## Contribuciones

1. Formulación con equivarianza exacta al flip, identidad a γ = 0 y cota de Lipschitz (§ Método).
2. Implementación con 25 pruebas unitarias y exportación ONNX verificada (ida y vuelta, tolerancia 1e-4).
3. Protocolo preregistrado con enmiendas E1–E7 y hash SHA-256, evaluación con 13 modelos/variantes.
4. Informe transparente de un resultado negativo parcial.
