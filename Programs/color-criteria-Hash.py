#!/usr/bin/env python3
"""
Script para análisis de PNe en HASH-IPHAS diferenciando T/P/L
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Configuración de criterios científicos
CRITERIA_PARAMS = {
    'c1': {'a': -0.5392, 'b': 1.8268},   # W1-W2 vs r-ha
    'c3': {'a': 1.6563, 'b': 0.7335},    # J-H vs H-W2
    'c5': {'a': 1.1100, 'b': 2.2339},    # J-W2 vs J-H
    'c7': {'a': 1.1098, 'b': 0.8292}     # J-K vs J-W1
}

# Configuración visual para T/P/L
PNSTAT_STYLES = {
    'T': {'color': '#27AE60', 'marker': 'o', 'edgecolor': '#145A32', 'label': 'Confirmadas (T)'},
    'P': {'color': '#F39C12', 'marker': 's', 'edgecolor': '#B9770E', 'label': 'Probables (P)'},
    'L': {'color': '#5DADE2', 'marker': 'D', 'edgecolor': '#21618C', 'label': 'Posibles (L)'}
}

OUTLIER_STYLE = {
    'marker': 'X',
    's': 80,
    'edgecolor': '#E74C3C',
    'linewidth': 1.5,
    'alpha': 1.0
}

def load_data(file_path):
    """Carga y limpia los datos manteniendo columnas esenciales"""
    required_cols = [
        'PNstat', 'r', 'ha', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag',
        'MajDiam', 'rErr', 'haErr', 'e_Jmag', 'e_Hmag', 'e_Kmag', 'e_W1mag', 'e_W2mag'
    ]
    
    df = pd.read_csv(
        file_path,
        usecols=required_cols,
        dtype={'PNstat': 'category'},
        na_values=['', 'NA', 'nan', 'NaN']
    )
    
    # Limpieza básica
    return df.dropna(subset=['PNstat', 'MajDiam']).copy()

def apply_quality_filters(df):
    """Aplica filtros de calidad fotométrica"""
    err_limits = {
        'rErr': 0.2,
        'haErr': 0.2,
        'e_Jmag': 0.2,
        'e_Hmag': 0.2,
        'e_Kmag': 0.2,
        'e_W1mag': 0.5,
        'e_W2mag': 0.5
    }
    
    mask = pd.Series(True, index=df.index)
    for col, max_err in err_limits.items():
        mask &= (df[col] <= max_err)
    
    return df[mask].copy()

def calculate_colors(df):
    """Calcula los colores necesarios para los diagramas"""
    return {
        'c1_x': df.W1mag - df.W2mag,
        'c1_y': df.r - df.ha,
        'c3_x': df.Jmag - df.Hmag,
        'c3_y': df.Hmag - df.W2mag,
        'c5_x': df.Jmag - df.Hmag,
        'c5_y': df.Jmag - df.W2mag,
        'c7_x': df.Jmag - df.Kmag,
        'c7_y': df.Jmag - df.W1mag
    }

def plot_diagrams(df, colors, output_file):
    """Genera diagramas color-color con identificación clara de T/P/L"""
    diagram_configs = [
        {
            'x': 'c1_x', 'y': 'c1_y', 'criteria': 'c1',
            'xlabel': 'W1 - W2 (mag)', 'ylabel': 'r - Ha (mag)',
            'title': 'Criterio 1: W1-W2 vs r-Hα'
        },
        {
            'x': 'c3_x', 'y': 'c3_y', 'criteria': 'c3',
            'xlabel': 'J - H (mag)', 'ylabel': 'H - W2 (mag)',
            'title': 'Criterio 3: J-H vs H-W2'
        },
        {
            'x': 'c5_x', 'y': 'c5_y', 'criteria': 'c5',
            'xlabel': 'J - H (mag)', 'ylabel': 'J - W2 (mag)',
            'title': 'Criterio 5: J-H vs J-W2'
        },
        {
            'x': 'c7_x', 'y': 'c7_y', 'criteria': 'c7',
            'xlabel': 'J - K (mag)', 'ylabel': 'J - W1 (mag)',
            'title': 'Criterio 7: J-K vs J-W1'
        }
    ]

    with PdfPages(output_file) as pdf:
        for config in diagram_configs:
            fig, ax = plt.subplots(figsize=(13, 10))
            
            # Graficar cada categoría
            for pn_stat in ['T', 'P', 'L']:
                if pn_stat not in df.PNstat.cat.categories:
                    continue
                
                mask = (df.PNstat == pn_stat)
                style = PNSTAT_STYLES[pn_stat]
                
                # Tamaño proporcional al diámetro (ajustar factor según necesidad)
                sizes = df.MajDiam[mask] * 2
                
                # Puntos dentro del criterio
                ax.scatter(
                    colors[config['x']][mask],
                    colors[config['y']][mask],
                    c=style['color'],
                    s=sizes,
                    marker=style['marker'],
                    edgecolor=style['edgecolor'],
                    linewidth=0.8,
                    alpha=0.7,
                    label=style['label']
                )
                
                # Puntos fuera del criterio
                a = CRITERIA_PARAMS[config['criteria']]['a']
                b = CRITERIA_PARAMS[config['criteria']]['b']
                outlier_mask = mask & (colors[config['y']] < (a * colors[config['x']] + b))
                
                if outlier_mask.any():
                    ax.scatter(
                        colors[config['x']][outlier_mask],
                        colors[config['y']][outlier_mask],
                        **OUTLIER_STYLE,
                        facecolor=style['color'],
                        label=f'{pn_stat} - Fuera de criterio'
                    )
            
            # Línea de criterio
            x_vals = np.linspace(
                np.nanmin(colors[config['x']]), 
                np.nanmax(colors[config['x']]), 
                100
            )
            ax.plot(
                x_vals,
                a * x_vals + b,
                color='#34495E',
                linestyle='--',
                linewidth=2,
                label='Límite del criterio'
            )
            
            # Configuración del gráfico
            ax.set(
                xlabel=config['xlabel'],
                ylabel=config['ylabel'],
                title=f"{config['title']}\nDiámetro mayor: tamaño del marcador"
            )
            ax.grid(alpha=0.2, linestyle=':')
            
            # Leyenda unificada
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax.legend(
                unique_labels.values(),
                unique_labels.keys(),
                loc='upper right',
                frameon=True,
                framealpha=0.9,
                scatterpoints=1
            )
            
            # Estadísticas
            total = len(df)
            stats_text = []
            for pn_stat in ['T', 'P', 'L']:
                if pn_stat in df.PNstat.cat.categories:
                    stat_total = sum(df.PNstat == pn_stat)
                    stat_outliers = sum(outlier_mask & (df.PNstat == pn_stat))
                    stats_text.append(
                        f"{pn_stat}: {stat_outliers}/{stat_total} ({stat_outliers/stat_total:.0%})"
                    )
            
            ax.text(
                0.95, 0.15,
                '\n'.join(stats_text),
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

def main():
    input_file = "HASH-IPHAS2arcsec-wise3arcsec.csv"
    output_file = "HASH_PNe_Analysis.pdf"
    
    try:
        print("🚀 Procesando datos HASH...")
        
        # 1. Carga de datos
        df = load_data(input_file)
        if df.empty:
            raise ValueError("Archivo vacío o sin columnas requeridas")
        print(f"📂 Datos cargados: {len(df)} objetos")
        
        # 2. Filtrado de calidad
        df_filtered = apply_quality_filters(df)
        if df_filtered.empty:
            raise ValueError("⚠️ Todos los datos eliminados por filtros de calidad")
        print(f"🔍 Datos filtrados: {len(df_filtered)} objetos")
        
        # 3. Configurar categorías
        df_filtered.PNstat = df_filtered.PNstat.cat.remove_unused_categories()
        
        # 4. Cálculo de colores
        colors = calculate_colors(df_filtered)
        
        # 5. Generar gráficos
        plot_diagrams(df_filtered, colors, output_file)
        print(f"\n🎉 Análisis completo! Resultados en: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if 'df' in locals():
            print("ℹ️ Columnas presentes:", df.columns.tolist())

if __name__ == "__main__":
    main()
