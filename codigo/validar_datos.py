"""
================================================================================
SCRIPT DE VALIDACIÓN DE DATOS
================================================================================

PROPÓSITO:
    Verificar que los datos están correctamente organizados ANTES de ejecutar
    el pipeline completo. Identifica problemas temprano.

USO:
    python validar_datos.py

OUTPUT:
    Reporte detallado de validación con recomendaciones
================================================================================
"""

import os
import cv2
import pandas as pd
import ast
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CSV_PATH = "data (1).csv"
IMAGE_DIR = "images"

# ============================================================================
# CLASE DE VALIDACIÓN
# ============================================================================

class DataValidator:
    """Valida la integridad del dataset antes del procesamiento."""
    
    def __init__(self, csv_path: str, image_dir: str):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.df = None
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def validate_csv_exists(self) -> bool:
        """Verifica que el CSV existe."""
        if not os.path.exists(self.csv_path):
            self.errors.append(f"CSV no encontrado: {self.csv_path}")
            return False
        return True
    
    def validate_image_dir_exists(self) -> bool:
        """Verifica que la carpeta de imágenes existe."""
        if not os.path.exists(self.image_dir):
            self.errors.append(f"Carpeta de imágenes no encontrada: {self.image_dir}")
            return False
        return True
    
    def load_csv(self) -> bool:
        """Carga el CSV."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.stats['total_rows'] = len(self.df)
            return True
        except Exception as e:
            self.errors.append(f"Error cargando CSV: {e}")
            return False
    
    def validate_csv_structure(self) -> bool:
        """Verifica estructura básica del CSV."""
        if self.df is None:
            return False
        
        required_columns = ['fname', 'target', 'bbox', 'image_height', 'image_width', 'source']
        missing = [col for col in required_columns if col not in self.df.columns]
        
        if missing:
            self.errors.append(f"Columnas faltantes en CSV: {missing}")
            return False
        
        print("✓ Estructura CSV válida")
        print(f"  Columnas: {list(self.df.columns)}")
        return True
    
    def validate_tb_filter(self) -> bool:
        """Valida datos de TB."""
        df_tb = self.df[self.df['target'] == 'tb']
        
        self.stats['tb_rows'] = len(df_tb)
        self.stats['tb_images'] = df_tb['fname'].nunique()
        
        if len(df_tb) == 0:
            self.warnings.append("No hay registros con target='tb'")
            return False
        
        print(f"✓ Registros TB encontrados")
        print(f"  Registros totales TB: {len(df_tb)}")
        print(f"  Imágenes únicas TB: {df_tb['fname'].nunique()}")
        
        # Imágenes con múltiples bboxes
        multi_bbox = (df_tb['fname'].value_counts() > 1).sum()
        self.stats['multi_bbox_images'] = multi_bbox
        print(f"  Imágenes con múltiples lesiones: {multi_bbox}")
        
        return True
    
    def validate_source_column(self) -> bool:
        """Valida valores en columna 'source'."""
        unique_sources = self.df['source'].unique()
        print(f"\n✓ Valores en 'source': {list(unique_sources)}")
        
        expected_sources = ['train', 'val', 'test']
        unexpected = [s for s in unique_sources if s not in expected_sources]
        
        if unexpected:
            self.warnings.append(f"Valores inesperados en 'source': {unexpected}")
        
        counts = self.df['source'].value_counts()
        print(f"  Distribución:")
        for source, count in counts.items():
            print(f"    - {source}: {count} ({100*count/len(self.df):.1f}%)")
        
        return True
    
    def validate_images_exist(self) -> Dict[str, int]:
        """Verifica que las imágenes existen."""
        df_tb = self.df[self.df['target'] == 'tb']
        unique_images = df_tb['fname'].unique()
        
        found = 0
        not_found = []
        
        for fname in unique_images:
            image_path = os.path.join(self.image_dir, fname)
            if os.path.exists(image_path):
                found += 1
            else:
                not_found.append(fname)
        
        self.stats['images_found'] = found
        self.stats['images_not_found'] = len(not_found)
        
        print(f"\n✓ Verificación de imágenes")
        print(f"  Encontradas: {found}/{len(unique_images)}")
        
        if not_found:
            self.warnings.append(f"Imágenes no encontradas: {len(not_found)}")
            print(f"  ⚠️  No encontradas: {len(not_found)}")
            print(f"     Primeras 5: {not_found[:5]}")
        
        return {'found': found, 'not_found': len(not_found)}
    
    def validate_image_dimensions(self) -> bool:
        """Valida que las dimensiones de imágenes coincidan con CSV."""
        df_tb = self.df[self.df['target'] == 'tb'].copy()
        unique_images = df_tb[['fname', 'image_height', 'image_width']].drop_duplicates()
        
        mismatches = []
        
        for _, row in unique_images.iterrows():
            fname = row['fname']
            csv_height = row['image_height']
            csv_width = row['image_width']
            
            image_path = os.path.join(self.image_dir, fname)
            if not os.path.exists(image_path):
                continue
            
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    mismatches.append((fname, "No se pudo cargar imagen"))
                    continue
                
                real_height, real_width = image.shape
                
                if real_height != csv_height or real_width != csv_width:
                    mismatches.append(
                        (fname, f"CSV: {csv_height}x{csv_width}, Real: {real_height}x{real_width}")
                    )
            except Exception as e:
                mismatches.append((fname, str(e)))
        
        if mismatches:
            self.warnings.append(f"Discrepancias de dimensiones: {len(mismatches)}")
            print(f"\n⚠️  Discrepancias de dimensiones: {len(mismatches)}")
            for fname, msg in mismatches[:5]:
                print(f"    - {fname}: {msg}")
            if len(mismatches) > 5:
                print(f"    ... y {len(mismatches)-5} más")
        else:
            print(f"\n✓ Dimensiones válidas")
        
        return len(mismatches) == 0
    
    def validate_bboxes(self) -> Dict:
        """Valida formato y validez de bounding boxes."""
        df_tb = self.df[self.df['target'] == 'tb'].copy()
        
        stats = {
            'total_bboxes': 0,
            'valid_bboxes': 0,
            'invalid_format': 0,
            'out_of_bounds': 0,
            'negative_dims': 0
        }
        
        invalid_examples = []
        
        for _, row in df_tb.iterrows():
            stats['total_bboxes'] += 1
            
            bbox_str = row['bbox']
            fname = row['fname']
            height = row['image_height']
            width = row['image_width']
            
            # Intentar parsear
            try:
                bbox_dict = ast.literal_eval(bbox_str)
            except:
                stats['invalid_format'] += 1
                invalid_examples.append((fname, "Format inválido"))
                continue
            
            # Validar claves
            required_keys = {'xmin', 'ymin', 'width', 'height'}
            if not required_keys.issubset(bbox_dict.keys()):
                stats['invalid_format'] += 1
                invalid_examples.append((fname, "Claves faltantes"))
                continue
            
            xmin = bbox_dict.get('xmin', 0)
            ymin = bbox_dict.get('ymin', 0)
            w = bbox_dict.get('width', 0)
            h = bbox_dict.get('height', 0)
            
            # Validar dimensiones
            if w <= 0 or h <= 0:
                stats['negative_dims'] += 1
                invalid_examples.append((fname, f"Dimensiones negativas: w={w}, h={h}"))
                continue
            
            xmax = xmin + w
            ymax = ymin + h
            
            # Validar límites
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                stats['out_of_bounds'] += 1
                invalid_examples.append(
                    (fname, f"Fuera de límites: ({xmin},{ymin})-({xmax},{ymax}) en {height}x{width}")
                )
                continue
            
            stats['valid_bboxes'] += 1
        
        self.stats['bbox_stats'] = stats
        
        print(f"\n✓ Validación de bounding boxes")
        print(f"  Total: {stats['total_bboxes']}")
        print(f"  Válidos: {stats['valid_bboxes']} ({100*stats['valid_bboxes']/stats['total_bboxes']:.1f}%)")
        print(f"  Inválidos:")
        print(f"    - Formato incorrecto: {stats['invalid_format']}")
        print(f"    - Dimensiones negativas: {stats['negative_dims']}")
        print(f"    - Fuera de límites: {stats['out_of_bounds']}")
        
        if invalid_examples:
            print(f"\n  Ejemplos de errores:")
            for fname, error in invalid_examples[:5]:
                print(f"    - {fname}: {error}")
            if len(invalid_examples) > 5:
                print(f"    ... y {len(invalid_examples)-5} más")
        
        return stats
    
    def check_image_quality(self) -> None:
        """Verifica calidad de imágenes (muestreo)."""
        df_tb = self.df[self.df['target'] == 'tb']
        sample_images = df_tb['fname'].unique()[:10]
        
        print(f"\n✓ Verificación de calidad de imágenes (muestra de {len(sample_images)})")
        
        for fname in sample_images:
            image_path = os.path.join(self.image_dir, fname)
            if not os.path.exists(image_path):
                continue
            
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"  ⚠️  {fname}: No se pudo leer")
                    continue
                
                h, w = image.shape
                min_val = image.min()
                max_val = image.max()
                mean_val = image.mean()
                
                print(f"  ✓ {fname}: {h}x{w}, intensidad [{min_val}-{max_val}], media={mean_val:.1f}")
                
            except Exception as e:
                print(f"  ✗ {fname}: Error {e}")
    
    def generate_report(self) -> str:
        """Genera reporte de validación."""
        report = "\n" + "="*80 + "\n"
        report += "REPORTE DE VALIDACIÓN DE DATOS\n"
        report += "="*80 + "\n\n"
        
        if self.errors:
            report += "ERRORES CRÍTICOS:\n"
            for error in self.errors:
                report += f"  ✗ {error}\n"
            report += "\n⚠️  El pipeline no se ejecutará correctamente.\n"
            report += "    Por favor, soluciona estos problemas primero.\n\n"
        else:
            report += "✓ Sin errores críticos\n\n"
        
        if self.warnings:
            report += "ADVERTENCIAS:\n"
            for warning in self.warnings:
                report += f"  ⚠️  {warning}\n"
            report += "\n"
        
        report += "ESTADÍSTICAS:\n"
        for key, value in self.stats.items():
            if isinstance(value, dict):
                report += f"  {key}:\n"
                for k, v in value.items():
                    report += f"    - {k}: {v}\n"
            else:
                report += f"  {key}: {value}\n"
        
        report += "\nRECOMENDACIONES:\n"
        
        if len(self.errors) > 0:
            report += "  1. Soluciona los errores críticos arriba\n"
        
        if 'images_not_found' in self.stats and self.stats['images_not_found'] > 0:
            report += "  2. Verifica que IMAGE_DIR apunta a la carpeta correcta\n"
            report += "     y que los nombres de archivos coinciden exactamente\n"
        
        if 'bbox_stats' in self.stats:
            bbox_stats = self.stats['bbox_stats']
            if bbox_stats['invalid_format'] > 0:
                report += "  3. Revisa el formato de las bounding boxes\n"
            if bbox_stats['out_of_bounds'] > 0:
                report += "  4. Algunas bboxes están fuera de los límites de la imagen\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report
    
    def run_all_validations(self):
        """Ejecuta todas las validaciones."""
        print("\n" + "="*80)
        print("INICIANDO VALIDACIÓN DE DATOS")
        print("="*80 + "\n")
        
        # Validaciones básicas
        if not self.validate_csv_exists():
            return
        
        if not self.validate_image_dir_exists():
            return
        
        if not self.load_csv():
            return
        
        print(f"\n✓ CSV cargado: {self.stats['total_rows']} registros\n")
        
        # Validaciones de contenido
        self.validate_csv_structure()
        self.validate_source_column()
        self.validate_tb_filter()
        
        if self.stats.get('tb_images', 0) > 0:
            self.validate_images_exist()
            self.validate_image_dimensions()
            self.validate_bboxes()
            self.check_image_quality()
        
        # Generar reporte
        report = self.generate_report()
        print(report)
        
        # Guardar reporte
        with open('validation_report.txt', 'w') as f:
            f.write(report)
        print("Reporte guardado en: validation_report.txt\n")
        
        return self.errors, self.warnings


# ============================================================================
# MAIN
# ============================================================================

def main():
    validator = DataValidator(CSV_PATH, IMAGE_DIR)
    validator.run_all_validations()


if __name__ == "__main__":
    main()
