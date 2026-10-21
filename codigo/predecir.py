"""
================================================================================
SCRIPT DE PREDICCIÓN: USAR MODELO ENTRENADO EN NUEVAS RADIOGRAFÍAS
================================================================================

PROPÓSITO:
    Cargar modelo U-Net entrenado y hacer predicciones en nuevas radiografías
    o en el dataset de validación. Genera visualizaciones y estadísticas.

USO:
    python predecir.py --model model_unet_tb.pth --image radiografia.png
    python predecir.py --model model_unet_tb.pth --dataset npz_dataset/val

================================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from tqdm import tqdm

# Importar modelo (asumiendo que está en el mismo directorio)
from ejemplo_train_unet import UNet2D

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = "model_unet_tb.pth"
CONFIDENCE_THRESHOLD = 0.5

print(f"Dispositivo: {DEVICE}")

# ============================================================================
# CLASE PREDICTOR
# ============================================================================

class TBPredictor:
    """Realiza predicciones de segmentación de TB."""
    
    def __init__(self, model_path, device=DEVICE):
        """Carga el modelo entrenado."""
        self.device = device
        self.model = UNet2D(in_channels=1, out_channels=2, init_features=64)
        self.model = self.model.to(device)
        
        # Cargar pesos
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Modelo cargado: {model_path}")
        else:
            print(f"❌ Modelo no encontrado: {model_path}")
            raise FileNotFoundError(f"No se encuentra {model_path}")
        
        self.model.eval()
    
    def predict_from_file(self, image_path):
        """
        Realiza predicción a partir de archivo de imagen.
        
        Args:
            image_path: Ruta a archivo PNG/JPG
            
        Returns:
            dict con imagen, predicción y confianza
        """
        # Cargar imagen
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ No se pudo cargar imagen: {image_path}")
            return None
        
        # Predicción
        pred_mask, confidence = self._predict_internal(image)
        
        return {
            'image': image,
            'mask': pred_mask,
            'confidence': confidence,
            'path': image_path,
            'shape': image.shape
        }
    
    def predict_from_npz(self, npz_path):
        """
        Realiza predicción a partir de archivo NPZ.
        
        Args:
            npz_path: Ruta a archivo .npz
            
        Returns:
            dict con predicción y comparación
        """
        # Cargar NPZ
        data = np.load(npz_path)
        image = data['image']
        mask_true = data['mask']
        
        # Predicción
        pred_mask, confidence = self._predict_internal(image)
        
        # Calcular métricas
        metrics = self._calculate_metrics(pred_mask, mask_true)
        
        return {
            'image': image,
            'mask_true': mask_true,
            'mask_pred': pred_mask,
            'confidence': confidence,
            'path': npz_path,
            'metrics': metrics
        }
    
    def _predict_internal(self, image_array):
        """
        Predicción interna (núcleo de predicción).
        
        Args:
            image_array: Array numpy (H, W) en escala grises
            
        Returns:
            (máscara_predicha, confianza_promedio)
        """
        # Normalizar
        image_norm = image_array.astype(np.float32) / 255.0
        
        # Convertir a tensor
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Predicción
        with torch.no_grad():
            output = self.model(image_tensor)
            prob = F.softmax(output, dim=1)
            
            # Probabilidad de lesión (clase 1)
            prob_lesion = prob[0, 1]  # (512, 512)
            
            # Máscara binaria
            mask_pred = (prob_lesion > CONFIDENCE_THRESHOLD).float().cpu().numpy()
        
        # Confianza promedio en región predicha
        confidence = float(prob_lesion[mask_pred == 1].mean()) if mask_pred.sum() > 0 else 0
        
        return mask_pred.astype(np.uint8), confidence
    
    def _calculate_metrics(self, pred_mask, true_mask):
        """Calcula métricas de predicción."""
        pred_f = torch.from_numpy(pred_mask).float()
        true_f = torch.from_numpy(true_mask).float()
        
        # Dice
        intersection = (pred_f * true_f).sum()
        dice = (2 * intersection) / (pred_f.sum() + true_f.sum() + 1e-6)
        
        # IoU
        union = (pred_f + true_f - pred_f * true_f).sum()
        iou = intersection / (union + 1e-6)
        
        # Sensitivity
        tp = (pred_f * true_f).sum()
        fn = ((1 - pred_f) * true_f).sum()
        sensitivity = tp / (tp + fn + 1e-6)
        
        # Specificity
        tn = ((1 - pred_f) * (1 - true_f)).sum()
        fp = (pred_f * (1 - true_f)).sum()
        specificity = tn / (tn + fp + 1e-6)
        
        return {
            'dice': float(dice),
            'iou': float(iou),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity)
        }


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def visualize_prediction(pred_result, save_path=None):
    """Visualiza resultado de predicción."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    image = pred_result['image']
    mask_pred = pred_result['mask_pred']
    confidence = pred_result['confidence']
    
    # Panel 1: Imagen original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('1. Imagen Original')
    axes[0, 0].axis('off')
    
    # Panel 2: Predicción
    axes[0, 1].imshow(mask_pred, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 1].set_title(f'2. Predicción\n(Confianza: {confidence:.3f})')
    axes[0, 1].axis('off')
    
    # Panel 3: Imagen + predicción
    axes[0, 2].imshow(image, cmap='gray')
    axes[0, 2].imshow(mask_pred, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
    axes[0, 2].set_title('3. Imagen + Predicción')
    axes[0, 2].axis('off')
    
    # Si hay ground truth, mostrar comparación
    if 'mask_true' in pred_result:
        mask_true = pred_result['mask_true']
        metrics = pred_result['metrics']
        
        # Panel 4: Ground truth
        axes[1, 0].imshow(mask_true, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 0].set_title('4. Ground Truth')
        axes[1, 0].axis('off')
        
        # Panel 5: Diferencia
        diff = np.abs(mask_pred - mask_true)
        axes[1, 1].imshow(diff, cmap='viridis')
        axes[1, 1].set_title('5. Diferencia')
        axes[1, 1].axis('off')
        
        # Panel 6: Métricas
        axes[1, 2].axis('off')
        metrics_text = f"""
        MÉTRICAS:
        
        Dice:           {metrics['dice']:.4f}
        IoU:            {metrics['iou']:.4f}
        Sensitivity:    {metrics['sensitivity']:.4f}
        Specificity:    {metrics['specificity']:.4f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                       verticalalignment='center')
    else:
        # Sin ground truth, mostrar información
        axes[1, 0].imshow(image, cmap='gray', alpha=0.5)
        axes[1, 0].imshow(mask_pred, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[1, 0].set_title('4. Overlay')
        axes[1, 0].axis('off')
        
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
    
    title = os.path.basename(pred_result['path'])
    fig.suptitle(f'Predicción TB: {title}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualización guardada: {save_path}")
    
    plt.show()


# ============================================================================
# PREDICCIÓN EN BATCH
# ============================================================================

def predict_dataset(predictor, dataset_dir, output_dir=None):
    """
    Realiza predicciones en todo un dataset.
    
    Args:
        predictor: Instancia de TBPredictor
        dataset_dir: Carpeta con archivos .npz
        output_dir: Carpeta para guardar resultados (opcional)
    """
    
    npz_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"❌ No se encontraron archivos .npz en {dataset_dir}")
        return
    
    print(f"\nRealizando predicciones en {len(npz_files)} imágenes...\n")
    
    results = []
    
    for npz_file in tqdm(npz_files, desc="Prediciendo"):
        npz_path = os.path.join(dataset_dir, npz_file)
        
        try:
            result = predictor.predict_from_npz(npz_path)
            results.append(result)
        except Exception as e:
            print(f"  ⚠️  Error en {npz_file}: {e}")
    
    # Estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE PREDICCIÓN")
    print("="*70 + "\n")
    
    if results:
        metrics_all = {
            'dice': [r['metrics']['dice'] for r in results],
            'iou': [r['metrics']['iou'] for r in results],
            'sensitivity': [r['metrics']['sensitivity'] for r in results],
            'specificity': [r['metrics']['specificity'] for r in results]
        }
        
        for metric_name, values in metrics_all.items():
            print(f"{metric_name.upper()}:")
            print(f"  Media:    {np.mean(values):.4f}")
            print(f"  Mediana:  {np.median(values):.4f}")
            print(f"  Std:      {np.std(values):.4f}")
            print(f"  Rango:    [{np.min(values):.4f}, {np.max(values):.4f}]")
            print()
    
    # Guardar resultados si se especifica
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar CSV con resultados
        csv_path = os.path.join(output_dir, 'predictions.csv')
        with open(csv_path, 'w') as f:
            f.write("filename,dice,iou,sensitivity,specificity\n")
            for result in results:
                fname = os.path.basename(result['path'])
                metrics = result['metrics']
                f.write(f"{fname},{metrics['dice']:.4f},{metrics['iou']:.4f},"
                       f"{metrics['sensitivity']:.4f},{metrics['specificity']:.4f}\n")
        
        print(f"✓ Resultados guardados: {csv_path}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Predicción con modelo U-Net entrenado'
    )
    parser.add_argument('--model', default=MODEL_CHECKPOINT, 
                       help='Ruta a modelo entrenado (.pth)')
    parser.add_argument('--image', help='Imagen individual para predicción')
    parser.add_argument('--dataset', help='Dataset (carpeta con .npz)')
    parser.add_argument('--output', help='Carpeta para guardar resultados')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PREDICCIÓN CON MODELO U-NET")
    print("="*70 + "\n")
    
    # Crear predictor
    predictor = TBPredictor(args.model, DEVICE)
    
    # Predicción individual
    if args.image:
        print(f"Prediciendo imagen: {args.image}\n")
        result = predictor.predict_from_file(args.image)
        
        if result:
            visualize_prediction(result)
            print(f"\n✓ Predicción completada")
            if result['mask'].sum() > 0:
                print(f"  Área predicha: {result['mask'].sum()} píxeles")
                print(f"  Confianza: {result['confidence']:.4f}")
            else:
                print(f"  Sin lesión detectada")
    
    # Predicción en dataset
    elif args.dataset:
        print(f"Dataset: {args.dataset}\n")
        results = predict_dataset(predictor, args.dataset, args.output)
        
        if results and len(results) > 0:
            # Visualizar algunos ejemplos
            print("\nVisualizando primeros 3 resultados...\n")
            for result in results[:3]:
                visualize_prediction(result)
    
    else:
        print("Proporcionar --image o --dataset")
        parser.print_help()


if __name__ == "__main__":
    main()
