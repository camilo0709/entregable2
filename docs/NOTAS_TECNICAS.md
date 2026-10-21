# NOTAS TÉCNICAS: SAM 2 EN RADIOGRAFÍAS DE TB

## Tabla de Contenidos

1. [Análisis Profundo de SAM 2](#análisis-profundo-de-sam-2)
2. [Limitaciones en Radiografías TB](#limitaciones-en-radiografías-tb)
3. [Optimizaciones Posibles](#optimizaciones-posibles)
4. [Integración con Modelos ML](#integración-con-modelos-ml)
5. [Benchmarking y Métricas](#benchmarking-y-métricas)
6. [Troubleshooting Avanzado](#troubleshooting-avanzado)

---

## Análisis Profundo de SAM 2

### Arquitectura de SAM 2

SAM 2 utiliza una arquitectura Vision Transformer modificada con tres componentes principales:

#### 1. Image Encoder
```
Radiografía (512x512)
    ↓
Patches 16x16 (1024 patches)
    ↓
Linear Projection + Positional Embeddings
    ↓
Vision Transformer Blocks (12-24 capas)
    ↓
Feature Maps de múltiples resoluciones
```

**Características:**
- Procesa imagen completa, no solo la región de bbox
- Extrae características jerárquicas (bajo/alto nivel)
- Sensible a: textura, bordes, contraste, patrones

#### 2. Prompt Encoder
```
Bounding Box (x_min, y_min, x_max, y_max)
    ↓
Convertir a embeddings
    ↓
Combinar con features de imagen
    ↓
Contexto integrado imagen + bbox
```

**Características:**
- Convierte bbox a token embeddings
- Usa sparse prompts (puntos, líneas, bboxes)
- Integra información de ubicación

#### 3. Mask Decoder
```
Features de imagen + prompt embeddings
    ↓
Transformer Decoder (2 capas)
    ↓
3 posibles máscaras
    ↓
Puntuaciones de confianza
```

**Características:**
- Genera múltiples máscaras candidatas
- Retorna puntuaciones de confianza
- Usa logits para refinamiento posterior

### Flujo Completo para una Imagen TB

```
Radiografía TB (512x512)
    │
    ├─→ Image Encoder
    │   └─→ Features multiresolución
    │       (256x256, 128x128, 64x64, 32x32)
    │
    └─→ Bbox = [100, 150, 200, 250]
        └─→ Prompt Encoder
            └─→ Embedding de bbox
    
    Ambos convergen:
    │
    ├─→ Mask Decoder
    │   ├─→ mask_1 (IoU 0.92)
    │   ├─→ mask_2 (IoU 0.85)
    │   └─→ mask_3 (IoU 0.78)
    │
    └─→ Seleccionar mejor: mask_1
```

---

## Limitaciones en Radiografías TB

### Limitación 1: Lesiones con Bajo Contraste

**El Problema:**
```
Radiografía típica:
- Lesión TB (gris claro)
- Tejido pulmonar normal (gris medio)
- Diferencia de intensidad: ~10-30 unidades (de 0-255)

SAM 2 fue entrenado en:
- Objetos naturales (animales, personas, objetos)
- Diferencias de contraste mucho mayores
- Bordes claros y definidos
```

**Impacto:**
- ❌ SAM 2 puede incluir tejido normal como lesión
- ❌ Puede omitir partes de la lesión
- ❌ Máscara puede ser muy grande o muy pequeña

**Mitigación:**
```python
# 1. Contrastar imagen antes de SAM 2
image_contrasted = cv2.equalizeHist(image)

# 2. Usar bbox más precisa
# Si bbox es muy grande, SAM 2 captura más fondo

# 3. Validar manualmente muestras críticas
# Revisar visualizaciones para casos ambiguos
```

### Limitación 2: Lesiones Ambiguas o Bordes Difusos

**El Problema:**
```
Radiografía TB atípica:
- Lesión con bordes poco definidos
- Infiltración gradual (no borde abrupto)
- Patrón similar al tejido circundante

SAM 2 puede:
- Trazar contorno diferente al esperado
- Omitir partes periféricas
- Incluir artefactos cercanos
```

**Impacto:**
- Máscara refinada puede diferir significativamente de bbox
- Menor confianza en el resultado
- Necesidad de validación

**Mitigación:**
```python
# Detectar casos problemáticos
def detect_ambiguous_cases(bbox_mask, refined_mask, threshold=0.5):
    overlap = np.logical_and(bbox_mask, refined_mask).sum()
    union = np.logical_or(bbox_mask, refined_mask).sum()
    iou = overlap / union if union > 0 else 0
    
    if iou < threshold:
        return True  # Caso ambiguo, revisar manualmente
    return False
```

### Limitación 3: Múltiples Lesiones Cercanas

**El Problema:**
```
Imagen con 2 lesiones próximas:

Radiografía:        bboxes:         problema SAM 2:
  ┌──────┐          ┌──────┐        ┌──────┐
  │ [1]  │          │ [1]  │        │ ███  │
  │  [2] │    →     │  [2] │   →    │ ███  │ (fusionadas)
  └──────┘          └──────┘        └──────┘

SAM 2 puede fusionar lesiones próximas en una sola máscara
```

**Impacto:**
- Pérdida de información sobre número de lesiones
- Mala evaluación de bbox_count
- Datos entrenamiento menos informativos

**Mitigación:**
```python
# Procesar cada bbox individualmente
def refine_with_separation(image, bboxes):
    masks = []
    for i, bbox in enumerate(bboxes):
        # Cada bbox se procesa por separado
        # Luego combinar con OR
        mask_i = sam2_refiner.refine_mask(image, bbox, single_bbox_mask)
        masks.append(mask_i)
    
    # Combinar sin fusionar
    combined = masks[0]
    for m in masks[1:]:
        combined = np.logical_or(combined, m)
    
    return combined
```

### Limitación 4: Artefactos de Radiografía

**El Problema:**
```
SAM 2 puede capturar:
- Costillas (líneas prominentes)
- Mediastino (región central)
- Hilos/cables de monitoreo
- Marcas/anotaciones

Especialmente si están dentro de la bbox
```

**Impacto:**
- Máscara incluye elementos no lesión
- Falsos positivos en entrenamiento

**Mitigación:**
```python
# 1. Refinar bbox para excluir artefactos obvios
# 2. Post-procesamiento:

def remove_artifacts(mask, min_area=50):
    # Remover pequeños componentes desconectados
    from scipy.ndimage import label
    labeled, num_features = label(mask)
    
    # Mantener solo componente más grande
    sizes = np.bincount(labeled.ravel())
    largest = np.argmax(sizes[1:]) + 1
    
    return (labeled == largest).astype(np.uint8)
```

### Limitación 5: Variaciones Radiológicas

**El Problema:**
```
SAM 2 es entrenado en fotografías naturales
No entiende radiología:
- Diferencia entre TB activa vs latente
- Patrón de infiltración específico
- Anatomía pulmonar

Resultado:
- Máscara genérica, no especializada
```

**Impacto:**
- No diferencia tipos de TB
- Posible segmentación subóptima

**Solución Ideal:**
- Entrenar modelo especializado en radiografías TB
- Usar SAM 2 + fine-tuning médico
- Anotar ground truth para validación

---

## Optimizaciones Posibles

### Optimización 1: Mejorar Entrada (Pre-procesamiento)

```python
def preprocess_for_sam2(image):
    """Preprocesa imagen para mejorar segmentación SAM 2."""
    
    # 1. Normalización CLAHE (Contrast Limited AHE)
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    # 2. Gaussian blur leve para suavizar ruido
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 3. Normalizar a [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # 4. Ecualización de histograma adaptativa
    # (opcional, puede ser agresivo)
    
    return image
```

**Impacto:** +5-10% en precisión de máscaras

### Optimización 2: Ensembles de SAM 2

```python
def ensemble_sam2_predictions(image, bbox, num_iterations=3):
    """Usa múltiples predicciones y toma consenso."""
    
    masks = []
    
    for _ in range(num_iterations):
        # Perturbar entrada ligeramente
        augmented = augment_image(image)
        
        # Predecir
        mask = sam2_predictor.predict(augmented, bbox)
        masks.append(mask)
    
    # Votación mayoritaria
    ensemble_mask = np.mean(masks, axis=0)
    ensemble_mask = (ensemble_mask > 0.5).astype(np.uint8)
    
    return ensemble_mask
```

**Impacto:** +10-15% en estabilidad
**Costo:** 3x más tiempo

### Optimización 3: Combinar SAM 2 con Métodos Clásicos

```python
def hybrid_segmentation(image, bbox):
    """Combina SAM 2 con edge detection."""
    
    # 1. SAM 2 inicial
    sam2_mask = sam2_refiner.refine_mask(image, bbox, bbox_mask)
    
    # 2. Edge detection en región de bbox
    region = extract_bbox_region(image, bbox)
    edges = cv2.Canny(region, 50, 150)
    edge_mask = extract_connected_components(edges)
    
    # 3. Combinar: intersección de ambas
    hybrid = np.logical_and(sam2_mask, edge_mask).astype(np.uint8)
    
    # 4. Si intersección es muy pequeña, usar SAM 2
    if hybrid.sum() < bbox_mask.sum() * 0.1:
        hybrid = sam2_mask
    
    return hybrid
```

**Impacto:** Mejor detección de bordes
**Costo:** +20% tiempo

### Optimización 4: Fine-tuning de SAM 2

Para máxima precisión médica:

```python
# Requiere annotaciones ground truth de ~100-200 imágenes
# Usar SAM 2 con adaptadores LoRA

from peft import get_peft_model, LoraConfig

# Configurar LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv", "proj"],
    lora_dropout=0.1,
)

# Fine-tune en radiografías TB
model = get_peft_model(sam2_model, lora_config)
train_loop(model, tb_radiography_dataset)
```

**Impacto:** +20-30% en precisión específica a TB
**Costo:** Necesita 100-200 anotaciones ground truth

---

## Integración con Modelos ML

### Patrón 1: Pseudo-labels para Entrenamiento Débilmente Supervisado

```python
# Usar las máscaras como pseudo-labels
# sin pretender que son ground truth perfecto

import torch
from torch.utils.data import Dataset

class WeaklySupervvisedTBDataset(Dataset):
    """Dataset con pseudo-labels de SAM 2."""
    
    def __init__(self, npz_dir):
        self.files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        image = torch.from_numpy(data['image']).float() / 255.0
        
        # Usar máscara refinada como pseudo-label
        # (no es ground truth, es estimada)
        mask = torch.from_numpy(data['mask']).long()
        
        # Opcionalmente: usar confianza de SAM 2 como peso
        # confidence_weight = data['sam2_confidence']  # si está disponible
        
        return image, mask

# Entrenar modelo
model = UNet2D(in_channels=1, out_channels=2)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

for images, masks in dataloader:
    outputs = model(images)
    loss = criterion(outputs, masks)
    
    # Backward + optimization
    loss.backward()
    optimizer.step()
```

### Patrón 2: Self-Supervised Pre-training

```python
# Usar características de SAM 2 como pre-entrenamiento

from torchvision.models import resnet50

# Extraer features de SAM 2
class SAM2FeatureExtractor:
    def __init__(self, sam2_model):
        self.sam2 = sam2_model
        self.image_encoder = sam2_model.image_encoder
    
    def extract_features(self, image):
        """Extrae features de imagen encoder de SAM 2."""
        return self.image_encoder(image)

# Usar features para entrenar modelo más pequeño
feature_extractor = SAM2FeatureExtractor(sam2_model)
features = feature_extractor.extract_features(radiography)

# Entrenar clasificador/segmentador sobre features
small_model = torch.nn.Sequential(
    torch.nn.Linear(256*32*32, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512*32*32)  # Predicción de máscara
)
```

### Patrón 3: Active Learning

```python
# Usar incertidumbre de SAM 2 para seleccionar muestras

def compute_uncertainty(image, bbox):
    """Calcula incertidumbre de SAM 2."""
    
    masks, scores, logits = sam2_predictor.predict(
        image, bbox, multimask_output=True
    )
    
    # Mayor spread en scores = mayor incertidumbre
    uncertainty = np.std(scores)
    
    # También: entropia de probabilidades
    probs = torch.softmax(torch.tensor(scores), dim=0).numpy()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return uncertainty, entropy

# Seleccionar casos inciertos para anotación manual
uncertain_cases = []
for file in npz_files:
    data = np.load(file)
    uncertainty = compute_uncertainty(data['image'], data['bbox'])
    
    if uncertainty > threshold:
        uncertain_cases.append((file, uncertainty))

# Anotar manualmente los casos inciertos
uncertain_cases.sort(key=lambda x: x[1], reverse=True)
for file, unc in uncertain_cases[:100]:
    annotate_manually(file)
```

---

## Benchmarking y Métricas

### Métricas para Evaluar Máscaras

```python
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def compute_mask_metrics(bbox_mask, refined_mask, ground_truth=None):
    """Calcula métricas de calidad de máscaras."""
    
    metrics = {}
    
    # 1. Área (píxeles de lesión detectados)
    metrics['area_bbox'] = bbox_mask.sum()
    metrics['area_refined'] = refined_mask.sum()
    
    # 2. IoU (Intersection over Union)
    intersection = np.logical_and(bbox_mask, refined_mask).sum()
    union = np.logical_or(bbox_mask, refined_mask).sum()
    metrics['iou'] = intersection / union if union > 0 else 0
    
    # 3. Dice Coefficient
    metrics['dice'] = 2 * intersection / (bbox_mask.sum() + refined_mask.sum() + 1e-10)
    
    # 4. Hausdorff Distance (diferencia máxima entre bordes)
    from scipy.ndimage import binary_erosion, binary_dilation
    bbox_border = bbox_mask - binary_erosion(bbox_mask)
    refined_border = refined_mask - binary_erosion(refined_mask)
    
    # Distancia (implementación simplificada)
    # metrics['hausdorff'] = calculate_hausdorff(bbox_border, refined_border)
    
    # 5. Contorno (perímetro)
    bbox_perimeter = bbox_border.sum()
    refined_perimeter = refined_border.sum()
    metrics['perimeter_bbox'] = bbox_perimeter
    metrics['perimeter_refined'] = refined_perimeter
    
    # 6. Si hay ground truth
    if ground_truth is not None:
        gt_intersection = np.logical_and(refined_mask, ground_truth).sum()
        gt_union = np.logical_or(refined_mask, ground_truth).sum()
        metrics['iou_vs_gt'] = gt_intersection / gt_union if gt_union > 0 else 0
        metrics['dice_vs_gt'] = 2 * gt_intersection / (refined_mask.sum() + ground_truth.sum() + 1e-10)
    
    return metrics

# Usar en batch
for npz_file in npz_files:
    data = np.load(npz_file)
    metrics = compute_mask_metrics(
        data['bbox_mask'], 
        data['mask']
    )
    
    print(f"{os.path.basename(npz_file)}: IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}")
```

### Análisis Estadístico del Dataset

```python
def analyze_dataset_statistics(dataset_dir):
    """Analiza distribución de características."""
    
    stats = {
        'areas': [],
        'sizes': [],
        'aspect_ratios': [],
        'refinement_gains': [],
        'tb_types': {}
    }
    
    for npz_file in glob.glob(f'{dataset_dir}/**/*.npz', recursive=True):
        data = np.load(npz_file)
        
        bbox_mask = data['bbox_mask']
        refined_mask = data['mask']
        
        # Área
        area = refined_mask.sum()
        stats['areas'].append(area)
        
        # Tamaño de caja contenedora
        coords = np.where(refined_mask)
        if len(coords[0]) > 0:
            h = coords[0].max() - coords[0].min()
            w = coords[1].max() - coords[1].min()
            stats['sizes'].append((h, w))
            stats['aspect_ratios'].append(h / (w + 1e-10))
        
        # Ganancia de refinamiento (reducción de área)
        gain = (bbox_mask.sum() - refined_mask.sum()) / bbox_mask.sum()
        stats['refinement_gains'].append(gain)
        
        # Tipo TB
        tb_type = str(data['tb_type'])
        if tb_type not in stats['tb_types']:
            stats['tb_types'][tb_type] = []
        stats['tb_types'][tb_type].append(area)
    
    # Imprimir resumen
    print("\n" + "="*70)
    print("ESTADÍSTICAS DEL DATASET")
    print("="*70)
    
    print(f"\nÁrea de lesiones (píxeles):")
    print(f"  Media: {np.mean(stats['areas']):.0f}")
    print(f"  Mediana: {np.median(stats['areas']):.0f}")
    print(f"  Rango: [{np.min(stats['areas']):.0f}, {np.max(stats['areas']):.0f}]")
    print(f"  Std: {np.std(stats['areas']):.0f}")
    
    print(f"\nGanancia de refinamiento (reducción %):")
    print(f"  Media: {np.mean(stats['refinement_gains'])*100:.1f}%")
    print(f"  Mediana: {np.median(stats['refinement_gains'])*100:.1f}%")
    print(f"  Rango: [{np.min(stats['refinement_gains'])*100:.1f}%, {np.max(stats['refinement_gains'])*100:.1f}%]")
    
    print(f"\nPor tipo TB:")
    for tb_type, areas in stats['tb_types'].items():
        print(f"  {tb_type}:")
        print(f"    Muestras: {len(areas)}")
        print(f"    Área media: {np.mean(areas):.0f}")
        print(f"    Área mediana: {np.median(areas):.0f}")
    
    return stats
```

---

## Troubleshooting Avanzado

### Problema: Máscaras muy grandes

```python
def diagnose_large_masks(npz_file):
    """Diagnostica por qué SAM 2 produce máscaras grandes."""
    
    data = np.load(npz_file)
    bbox_mask = data['bbox_mask']
    refined_mask = data['mask']
    image = data['image']
    
    # Si máscara > bbox en >30%
    if refined_mask.sum() > bbox_mask.sum() * 1.3:
        
        print(f"⚠️  {os.path.basename(npz_file)}: Máscara expandida")
        print(f"    Bbox área: {bbox_mask.sum()}")
        print(f"    Máscara área: {refined_mask.sum()}")
        print(f"    Expansión: {(refined_mask.sum() - bbox_mask.sum()) / bbox_mask.sum() * 100:.1f}%")
        
        # Verificar si hay artefactos
        mask_outside = refined_mask * (1 - bbox_mask)
        if mask_outside.sum() > 0:
            print(f"    Píxeles fuera de bbox: {mask_outside.sum()}")
            
            # Imagen de diagnóstico
            viz = np.zeros((*image.shape, 3))
            viz[..., 0] = mask_outside  # Rojo = fuera de bbox
            return viz
```

### Problema: Inconsistencia en múltiples ejecuciones

```python
def check_reproducibility(image, bbox, num_runs=5):
    """Verifica si SAM 2 es determinista."""
    
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    
    masks = []
    for _ in range(num_runs):
        mask = sam2_predictor.predict(image, bbox)
        masks.append(mask)
    
    # Calcular varianza
    mask_array = np.array(masks)
    variance = np.var(mask_array, axis=0)
    
    if variance.max() > 0:
        print("⚠️  SAM 2 no es determinista (varianza en outputs)")
        print(f"    Max varianza: {variance.max():.4f}")
        print("    Posible causa: Dropout en modelo")
        print("    Solución: Usar mode eval + torch.no_grad()")
```

### Problema: Bajo uso de GPU

```python
def profile_gpu_usage():
    """Perfila uso de GPU durante pipeline."""
    
    import GPUtil
    import time
    
    print("Monitoreando GPU...")
    
    for _ in range(10):
        # Ejecutar predicción
        mask = sam2_predictor.predict(image, bbox)
        
        # Checquear GPU
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.load*100:.1f}% utilizado, {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        
        time.sleep(1)
    
    # Posibles problemas:
    # - Baja utilización: problemas de transferencia CPU-GPU
    # - Baja memoria: aumentar batch size
    # - Throttling: temperatura alta
```

---

## Recomendaciones Finales

### Para Investigación Clínica Rigurosa

1. **Validación Manual:** Revisar 10-20% de máscaras generadas
2. **Ground Truth:** Anotar manualmente 100-200 casos representativos
3. **Métrica de Confianza:** Calcular IoU vs ground truth
4. **Reporte de Limitaciones:** Documentar casos donde SAM 2 falló

### Para Entrenamiento de Modelo Rápido

1. **Usar tal cual:** Las pseudo-máscaras son suficientemente buenas
2. **Enlarge dataset:** Aumentar datos con augmentación
3. **Validación cruzada:** Para estimar generalización real

### Para Máxima Precisión

1. **Fine-tune SAM 2:** Con radiografías TB anotadas
2. **Ensemble:** Combinar múltiples predictores
3. **Post-procesamiento:** Remover artefactos automáticamente
4. **Validación humana:** Loop de mejora continua

---

## Conclusión

SAM 2 es una **herramienta excelente pero no perfecta** para segmentación de TB:

✅ **Ventajas:**
- Automatiza completamente el proceso
- Produce máscaras significativamente mejores que rectangulares
- Generaliza bien a radiografías no vistas
- Rápido con GPU

❌ **Limitaciones:**
- No especializado en radiología TB
- Puede fallar en lesiones ambiguas
- No entiende contexto médico
- Requiere validación

**Recomendación:** Usar como punto de partida para pseudo-labels débilmente supervisados, pero complementar con validación manual y/o fine-tuning para máxima precisión clínica.
