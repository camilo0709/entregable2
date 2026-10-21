# TBX11K · Benchmark de Deep Learning para la segmentación de lesiones de tuberculosis

**Entregable 2 — Deep Learning** · Equipo: Mujica Camilo y Romero Carlos · 2026

Este informe ejecuta un benchmark controlado de **10 arquitecturas × 5 semillas**
(2 baseline incluidos) sobre el dataset **TBX11K**, con un *ground truth* de
prueba congelado (plata, D1-R1) y un protocolo de inferencia estadística
predefinido. Cada capítulo cierra con decisiones explícitas, no con descripción.

## Estructura y ponderación (rúbrica)

| § | Contenido | Peso | Estado |
|---|---|---|---|
| 1 | [Problema, preguntas de investigación y dataset](01_planteamiento.md) | 15 % | ✅ formulación, 4 RQ con H₀/H₁, Data Card |
| 2 | [EDA orientado a decisiones](EDA_Decisiones_TBX11K.ipynb) | — | ✅ ejecutado (A1–A9) |
| 3 | [Revisión sistemática del estado del arte](03_revision_sistematica.md) | 15 % | ✅ protocolo, PRISMA, Top 10, gap |
| 4 | [Implementación de modelos benchmark](04_implementacion.md) | 25 % | 🚀 entrenando (Colab) |
| 5 | [Paper guía y replicación](05_paper_guia.md) | — | ✍️ en preparación |
| 6 | [Benchmark final (inferencia estadística)](06_benchmark_final.md) | 20 % | ⏳ con resultados §4 |
| 7 | [Ruta metodológica (modality I1–I8)](07_ruta_metodologica.md) | 15 % | ✍️ en preparación |
| 8 | [Reproducibilidad, ética y comunicación](08_reproducibilidad.md) | 10 % | ✍️ en preparación |

## Preguntas de investigación (RQ)

1. **RQ1** — ¿Existen diferencias *estadísticamente significativas* entre familias
   de arquitecturas (CNN, atención, transformador, híbridas, modernas) bajo
   presupuesto idéntico y múltiples semillas?
2. **RQ2** — ¿Los transformadores (y híbridos) superan a las CNN clásicas en baja
   señal (lesión ≈ 8.8 % de píxeles) o solo empatan a mayor coste?
3. **RQ3** — ¿Cuánto del error se debe al *techo del ground truth* y cuánto al modelo?
4. **RQ4** — ¿Grad-CAM coincide con la localización humana de la lesión
   (o los modelos aciertan por las razones equivocadas)?

## Cómo reproducir este libro

```bash
git clone <repo> && cd proyecto_tb
python -m venv .venv && . .venv/bin/activate
pip install -r libro/requirements.txt
jupyter-book build libro/          # ejecuta los notebooks y genera el HTML
open libro/_build/html/index.html
```

El build **ejecuta las celdas** en cada compilación
(`execute.execute_notebooks: force` en `_config.yml`): los resultados del HTML
siempre provienen de código corriendo en esa máquina, no de salidas pegadas.
Los notebooks también se abren directamente en Jupyter/Colab desde el repo.

## Datos y salvaguardas

- **Dataset:** TBX11K (11 200 CXR de DA/DB/Montgomery/Shenzhen, Liu et al., CVPR 2020); splits por paciente
  (`datos/splits/`, sellados por manifest SHA-256).
- **GT de prueba:** plata generado con SAM 2 y congelado (§2, decisión D1-R1);
  techo humano medido 0.9879. Los CSVs sellados jamás se modifican.
- **Control OOD:** `test_neg` (sanos de otra adquisición) se usa solo como
  control de falsos positivos (sesgo de adquisición detectado en el EDA: AUC 0.842).
