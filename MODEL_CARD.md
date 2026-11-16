# Model card · LACCA-UNet

**Autores:** Mujica Camilo y Romero Carlos.
**Tarea:** segmentación de lesiones de TB en radiografías de tórax (TBX11K), entrada 1×512×512, z-score por imagen, salida logits (sigmoide, umbral 0.5).
**Arquitectura:** U-Net con encoder ResNet34 y módulo LACCA en etapas 3 y 4.
**Rendimiento:** Dice de test (80 positivas, 10 semillas) 0.5651 ± 0.022; no superior a Attention U-Net (0.5595) ni U-Net++ (0.5574).
**Datos:** TBX11K con GT de plata (SAM2); sin anotación experta de máscaras.
**Limitaciones:** frágil ante ruido y desenfoque; calibración mejorable; no validado clínicamente; validación externa solo con máscaras pulmonares.
**Uso previsto:** investigación y docencia. **No usar** para diagnóstico clínico.
**Reproducibilidad:** ver `libro/book/e3_05_reproducibilidad.md`, `experiments/`, `artifacts/`.
