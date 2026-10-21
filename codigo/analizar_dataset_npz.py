"""
================================================================================
ANÁLISIS ESTADÍSTICO DEL DATASET NPZ
================================================================================

PROPÓSITO:
    Analiza estadísticas detalladas del dataset generado para comprender:
    - Distribución de tamaños de lesiones
    - Concordancia bbox vs máscara refinada
    - Variabilidad entre particiones
    - Comparación active vs latent TB

USO:
    python analizar_dataset_npz.py

OUTPUT:
    - Tabla de estadísticas
    - Gráficos de distribuciones
    - Reporte en CSV

================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATASET_DIR = "npz_dataset"
OUTPUT_DIR = "dataset_analysis"
PARTITIONS = ['train', 'val', 'test']

# ============================================================================
# ANALIZADOR
# ============================================================================

class DatasetAnalyzer:
    """Analiza estadísticas del dataset NPZ."""
    
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data = []
        self.stats = {}
    
    def load_dataset(self):
        """Carga todos los archivos NPZ."""
        print("Cargando dataset...\n")
        
        for partition in PARTITIONS:
            partition_dir = os.path.join(self.dataset_dir, partition)
            
            if not os.path.exists(partition_dir):
                continue
            
            npz_files = [f for f in os.listdir(partition_dir) if f.endswith('.npz')]
            
            print(f"  {partition}: {len(npz_files)} archivos")
            
            for npz_file in tqdm(npz_files, desc=f"    Analizando {partition}"):
                try:
                    npz_path = os.path.join(partition_dir, npz_file)
                    data = np.load(npz_path)
                    
                    self.data.append({
                        'partition': partition,
                        'filename': npz_file,
                        'tb_type': str(data['tb_type']),
                        'bbox_count': int(data['bbox_count']),
                        'image_mean': float(data['image'].mean()),
                        'image_std': float(data['image'].std()),
                        'bbox_mask_area': int(data['bbox_mask'].sum()),
                        'mask_area': int(data['mask'].sum()),
                        'image_shape': data['image'].shape,
                        'has_lesion': int(data['mask'].sum() > 0)
                    })
                
                except Exception as e:
                    print(f"      ⚠️  Error en {npz_file}: {e}")
        
        print(f"\n✓ Total de muestras cargadas: {len(self.data)}\n")
        
        return pd.DataFrame(self.data)
    
    def compute_statistics(self, df):
        """Calcula estadísticas descriptivas."""
        print("="*70)
        print("ESTADÍSTICAS GENERALES")
        print("="*70 + "\n")
        
        print(f"Total de muestras: {len(df)}")
        print(f"Particiones:")
        for partition in PARTITIONS:
            count = len(df[df['partition'] == partition])
            if count > 0:
                print(f"  - {partition}: {count} ({100*count/len(df):.1f}%)")
        
        print(f"\n" + "-"*70)
        print("TIPOS DE TB")
        print("-"*70 + "\n")
        
        for tb_type in df['tb_type'].unique():
            subset = df[df['tb_type'] == tb_type]
            print(f"{tb_type}:")
            print(f"  Muestras: {len(subset)}")
            print(f"  % del total: {100*len(subset)/len(df):.1f}%")
        
        print(f"\n" + "-"*70)
        print("TAMAÑO DE LESIONES (píxeles)")
        print("-"*70 + "\n")
        
        for metric in ['bbox_mask_area', 'mask_area']:
            areas = df[metric]
            metric_name = "Pseudomáscara" if "bbox" in metric else "Máscara refinada"
            
            print(f"{metric_name}:")
            print(f"  Media:        {areas.mean():.0f}")
            print(f"  Mediana:      {areas.median():.0f}")
            print(f"  Std:          {areas.std():.0f}")
            print(f"  Mínimo:       {areas.min():.0f}")
            print(f"  Máximo:       {areas.max():.0f}")
            print(f"  Q1 (25%):     {areas.quantile(0.25):.0f}")
            print(f"  Q3 (75%):     {areas.quantile(0.75):.0f}")
            print()
        
        print("-"*70)
        print("MÚLTIPLES LESIONES")
        print("-"*70 + "\n")
        
        for count in sorted(df['bbox_count'].unique()):
            subset = df[df['bbox_count'] == count]
            print(f"  {count} lesión(es): {len(subset)} imágenes ({100*len(subset)/len(df):.1f}%)")
        
        print(f"\n" + "-"*70)
        print("REFINAMIENTO CON SAM 2")
        print("-"*70 + "\n")
        
        # Comparar bbox_mask vs mask
        df['area_ratio'] = df['mask_area'] / (df['bbox_mask_area'] + 1)
        df['area_reduction'] = (df['bbox_mask_area'] - df['mask_area']) / (df['bbox_mask_area'] + 1)
        
        print(f"Ratio area máscara/bbox (promedio): {df['area_ratio'].mean():.3f}")
        print(f"Reducción promedio del área: {df['area_reduction'].mean()*100:.1f}%")
        print(f"Rango de reducción: [{df['area_reduction'].min()*100:.1f}%, {df['area_reduction'].max()*100:.1f}%]")
        
        # Contar cuándo mask > bbox (anomalías)
        expanded = (df['mask_area'] > df['bbox_mask_area']).sum()
        print(f"\nCasos donde máscara > bbox: {expanded} ({100*expanded/len(df):.1f}%)")
        print(f"Casos donde máscara ≈ bbox: {(df['area_ratio'] > 0.9).sum()}")
        print(f"Casos donde máscara < bbox: {(df['area_ratio'] < 0.5).sum()}")
        
        return df
    
    def analyze_by_subgroup(self, df):
        """Analiza por subgrupos."""
        print("\n" + "="*70)
        print("ANÁLISIS POR SUBGRUPOS")
        print("="*70 + "\n")
        
        # Por tipo TB
        print("POR TIPO TB:")
        print("-"*70 + "\n")
        
        for tb_type in df['tb_type'].unique():
            subset = df[df['tb_type'] == tb_type]
            
            print(f"{tb_type}:")
            print(f"  Muestras: {len(subset)}")
            print(f"  Área promedio (mask): {subset['mask_area'].mean():.0f}")
            print(f"  Área promedio (bbox): {subset['bbox_mask_area'].mean():.0f}")
            print(f"  Lesiones múltiples: {(subset['bbox_count'] > 1).sum()}")
            print()
        
        # Por partición
        print("POR PARTICIÓN:")
        print("-"*70 + "\n")
        
        for partition in PARTITIONS:
            subset = df[df['partition'] == partition]
            if len(subset) == 0:
                continue
            
            print(f"{partition}:")
            print(f"  Muestras: {len(subset)}")
            print(f"  Área promedio (mask): {subset['mask_area'].mean():.0f}")
            print(f"  Lesiones múltiples: {(subset['bbox_count'] > 1).sum()}")
            print(f"  Sin lesión detectada: {(subset['mask_area'] == 0).sum()}")
            print()
    
    def generate_plots(self, df, output_dir):
        """Genera visualizaciones."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerando gráficos...\n")
        
        # Figura 1: Distribución de tamaños de lesiones
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(df['bbox_mask_area'], bins=50, alpha=0.7, label='BBox', color='red')
        axes[0].hist(df['mask_area'], bins=50, alpha=0.7, label='Refinada', color='blue')
        axes[0].set_xlabel('Área (píxeles)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Tamaños de Lesiones')
        axes[0].legend()
        axes[0].set_yscale('log')
        
        axes[1].scatter(df['bbox_mask_area'], df['mask_area'], alpha=0.5)
        axes[1].plot([0, df['bbox_mask_area'].max()], [0, df['bbox_mask_area'].max()], 
                    'k--', label='Identidad')
        axes[1].set_xlabel('Área BBox')
        axes[1].set_ylabel('Área Máscara Refinada')
        axes[1].set_title('Comparación de Áreas')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_distribucion_tamanios.png'), dpi=150)
        plt.close()
        print("  ✓ 01_distribucion_tamanios.png")
        
        # Figura 2: Por tipo TB
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        tb_types = df['tb_type'].unique()
        areas_by_type = [df[df['tb_type'] == tb]['mask_area'] for tb in tb_types]
        
        axes[0].boxplot(areas_by_type, labels=tb_types)
        axes[0].set_ylabel('Área (píxeles)')
        axes[0].set_title('Tamaño de Lesiones por Tipo TB')
        axes[0].grid(alpha=0.3)
        
        count_by_type = df['tb_type'].value_counts()
        axes[1].bar(count_by_type.index, count_by_type.values, color=['#1f77b4', '#ff7f0e'][:len(count_by_type)])
        axes[1].set_ylabel('Número de Imágenes')
        axes[1].set_title('Distribución de Tipos TB')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_por_tipo_tb.png'), dpi=150)
        plt.close()
        print("  ✓ 02_por_tipo_tb.png")
        
        # Figura 3: Por partición
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        partitions = [p for p in PARTITIONS if len(df[df['partition'] == p]) > 0]
        count_by_partition = [len(df[df['partition'] == p]) for p in partitions]
        
        axes[0].bar(partitions, count_by_partition, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Número de Imágenes')
        axes[0].set_title('Distribución por Partición')
        axes[0].grid(alpha=0.3, axis='y')
        
        # Añadir porcentajes
        for i, (p, c) in enumerate(zip(partitions, count_by_partition)):
            axes[0].text(i, c, f'{100*c/len(df):.1f}%', ha='center', va='bottom')
        
        # Box plot por partición
        areas_by_partition = [df[df['partition'] == p]['mask_area'] for p in partitions]
        axes[1].boxplot(areas_by_partition, labels=partitions)
        axes[1].set_ylabel('Área (píxeles)')
        axes[1].set_title('Tamaño de Lesiones por Partición')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_por_particion.png'), dpi=150)
        plt.close()
        print("  ✓ 03_por_particion.png")
        
        # Figura 4: Lesiones múltiples
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        bbox_counts = df['bbox_count'].value_counts().sort_index()
        axes[0].bar(bbox_counts.index, bbox_counts.values, color='coral', alpha=0.7)
        axes[0].set_xlabel('Número de Lesiones')
        axes[0].set_ylabel('Número de Imágenes')
        axes[0].set_title('Distribución de Múltiples Lesiones')
        axes[0].grid(alpha=0.3, axis='y')
        
        # Comparación de tamaño por número de lesiones
        single = df[df['bbox_count'] == 1]['mask_area']
        multi = df[df['bbox_count'] > 1]['mask_area']
        
        axes[1].boxplot([single, multi], labels=['1 lesión', 'Múltiples'])
        axes[1].set_ylabel('Área (píxeles)')
        axes[1].set_title('Tamaño de Lesiones: Simple vs Múltiple')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_lesiones_multiples.png'), dpi=150)
        plt.close()
        print("  ✓ 04_lesiones_multiples.png")
        
        # Figura 5: Refinamiento
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        reduction = (df['bbox_mask_area'] - df['mask_area']) / df['bbox_mask_area'] * 100
        
        axes[0].hist(reduction, bins=50, color='green', alpha=0.7)
        axes[0].set_xlabel('Reducción de Área (%)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Refinamiento por SAM 2')
        axes[0].axvline(reduction.mean(), color='red', linestyle='--', label=f'Media: {reduction.mean():.1f}%')
        axes[0].legend()
        
        axes[1].scatter(df['bbox_mask_area'], reduction, alpha=0.5)
        axes[1].set_xlabel('Área BBox (píxeles)')
        axes[1].set_ylabel('Reducción (%)')
        axes[1].set_title('Refinamiento vs Tamaño Original')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_refinamiento.png'), dpi=150)
        plt.close()
        print("  ✓ 05_refinamiento.png")
    
    def save_statistics(self, df, output_dir):
        """Guarda estadísticas en CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar dataframe completo
        csv_path = os.path.join(output_dir, 'estadisticas_completas.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Estadísticas guardadas: {csv_path}")
        
        # Resumen por partición
        summary = []
        for partition in PARTITIONS:
            subset = df[df['partition'] == partition]
            if len(subset) == 0:
                continue
            
            summary.append({
                'Partición': partition,
                'Muestras': len(subset),
                'Media Área': f"{subset['mask_area'].mean():.0f}",
                'Mediana Área': f"{subset['mask_area'].median():.0f}",
                'Std Área': f"{subset['mask_area'].std():.0f}",
                'Lesiones Múltiples': (subset['bbox_count'] > 1).sum(),
                'Sin Lesión': (subset['mask_area'] == 0).sum()
            })
        
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, 'resumen_particiones.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"  ✓ Resumen: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ANÁLISIS ESTADÍSTICO DEL DATASET NPZ")
    print("="*70 + "\n")
    
    # Crear analizador
    analyzer = DatasetAnalyzer(DATASET_DIR)
    
    # Cargar datos
    df = analyzer.load_dataset()
    
    if len(df) == 0:
        print("❌ No se encontraron datos")
        return
    
    # Análisis estadístico
    df = analyzer.compute_statistics(df)
    
    # Análisis por subgrupos
    analyzer.analyze_by_subgroup(df)
    
    # Generar gráficos
    analyzer.generate_plots(df, OUTPUT_DIR)
    
    # Guardar estadísticas
    analyzer.save_statistics(df, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print(f"✓ ANÁLISIS COMPLETO")
    print("="*70)
    print(f"\nResultados guardados en: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
