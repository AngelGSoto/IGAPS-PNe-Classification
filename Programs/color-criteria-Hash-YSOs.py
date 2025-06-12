#!/usr/bin/env python3
"""
Análisis comparativo PNe vs YSOs en diagramas color-color (versión mejorada)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# Configuración unificada de parámetros
CONFIG = {
    'criterios': {
        'W1W2_JH': {'a': 0.75, 'b': 0.85, 'type': 'vertical-horizontal', 'label': 'Corte principal'},
        'HW2_JH': {'a': 2.7, 'b': 0.85, 'type': 'vertical-horizontal', 'label': 'Corte principal'},
        'Akras_JH': {'value': 0.5, 'type': 'horizontal', 'style': '-.', 'color': '#C0392B', 'label': 'Akras et al. (2019)'},
        'Akras_HW2': {'value': 2.24, 'type': 'vertical', 'style': '-.', 'color': '#C0392B', 'label': 'Akras et al. (2019)'}
    },
    
    'estilos': {
        'PNe': {
            'T': {'color': '#27AE60', 'marker': 'o', 'label': 'PNe Confirmadas (T)', 'size': 80, 'zorder': 4},
            'P': {'color': '#F1C40F', 'marker': 's', 'label': 'PNe Probables (P)', 'size': 60, 'zorder': 3},
            'L': {'color': '#2980B9', 'marker': 'D', 'label': 'PNe Posibles (L)', 'size': 50, 'zorder': 2}
        },
        'YSOs': {
            'color': '#E74C3C', 
            'marker': 'X', 
            'label': 'YSOs (Prob ≥ 0.9)',
            'size': 45,
            'alpha': 0.7,
            'zorder': 1
        },
        'lineas': {
            'principal': {'color': '#2C3E50', 'linestyle': '--', 'linewidth': 2},
            'akras': {'color': '#C0392B', 'linestyle': '-.', 'linewidth': 1.5}
        }
    }
}

def load_data(file_path, obj_type):
    """Carga y valida los datos de entrada"""
    required_cols = {
        'PNe': ['PNstat', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'r', 'ha'],
        'YSOs': ['Prob', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'r', 'ha']
    }
    
    df = pd.read_csv(
        file_path,
        usecols=required_cols[obj_type],
        dtype={'PNstat': 'category'} if obj_type == 'PNe' else None,
        na_values=['', 'NA', 'nan', 'NaN']
    ).dropna()
    
    if obj_type == 'YSOs':
        df = df[df['Prob'] >= 0.9]
    
    return df

def calculate_colors(df):
    """Calcula los índices de color necesarios"""
    return {
        'J_H': df.Jmag - df.Hmag,
        'H_W2': df.Hmag - df.W2mag,
        'W1_W2': df.W1mag - df.W2mag,
        'J_W2': df.Jmag - df.W2mag,
        'r_ha': df.r - df.ha,
        'J_K': df.Jmag - df.Kmag,
        'J_W1': df.Jmag - df.W1mag
    }

def plot_diagrams(pne_df, yso_df, pne_colors, yso_colors, output_file):
    """Genera los diagramas color-color en PDF"""
    diagramas = [
        {
            'x': 'W1_W2', 'y': 'J_H',
            'lineas': ['W1W2_JH', 'Akras_JH'],
            'titulo': 'W1-W2 vs J-H',
            'xlabel': r'$W1 - W2$ (mag)',
            'ylabel': r'$J - H$ (mag)'
        },
        {
            'x': 'H_W2', 'y': 'J_H',
            'lineas': ['HW2_JH', 'Akras_HW2', 'Akras_JH'],
            'titulo': 'H-W2 vs J-H',
            'xlabel': r'$H - W2$ (mag)',
            'ylabel': r'$J - H$ (mag)'
        },
        {
            'x': 'J_K', 'y': 'J_W1',
            'lineas': [],
            'titulo': 'J-K vs J-W1',
            'xlabel': r'$J - K$ (mag)',
            'ylabel': r'$J - W1$ (mag)'
        }
    ]

    with PdfPages(output_file) as pdf:
        for diagrama in diagramas:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # ===== Graficar YSOs =====
            ax.scatter(
                yso_colors[diagrama['x']],
                yso_colors[diagrama['y']],
                c=CONFIG['estilos']['YSOs']['color'],
                s=CONFIG['estilos']['YSOs']['size'],
                marker=CONFIG['estilos']['YSOs']['marker'],
                alpha=CONFIG['estilos']['YSOs']['alpha'],
                label=CONFIG['estilos']['YSOs']['label'],
                zorder=CONFIG['estilos']['YSOs']['zorder']
            )
            
            # ===== Graficar PNe =====
            for tipo in ['T', 'P', 'L']:
                if tipo not in pne_df.PNstat.cat.categories:
                    continue
                
                mask = pne_df.PNstat == tipo
                estilo = CONFIG['estilos']['PNe'][tipo]
                
                ax.scatter(
                    pne_colors[diagrama['x']][mask],
                    pne_colors[diagrama['y']][mask],
                    c=estilo['color'],
                    s=estilo['size'],
                    marker=estilo['marker'],
                    edgecolor='k',
                    linewidth=0.8,
                    label=estilo['label'],
                    zorder=estilo['zorder']
                )
            
            # ===== Líneas de referencia =====
            line_handles = []
            for linea in diagrama['lineas']:
                params = CONFIG['criterios'][linea]
                
                if params['type'] == 'vertical':
                    ax.axvline(
                        params['value'],
                        color=params['color'],
                        linestyle=params['style'],
                        linewidth=params.get('linewidth', 1.5),
                        zorder=1
                    )
                    line_handles.append(
                        Line2D([], [], color=params['color'], 
                               linestyle=params['style'], 
                               label=params['label'])
                    )
                elif params['type'] == 'horizontal':
                    ax.axhline(
                        params['value'],
                        color=params['color'],
                        linestyle=params['style'],
                        linewidth=params.get('linewidth', 1.5),
                        zorder=1
                    )
                    line_handles.append(
                        Line2D([], [], color=params['color'], 
                               linestyle=params['style'], 
                               label=params['label'])
                    )
                elif params['type'] == 'vertical-horizontal':
                    ax.axvline(
                        params['a'],
                        color=CONFIG['estilos']['lineas']['principal']['color'],
                        linestyle=CONFIG['estilos']['lineas']['principal']['linestyle'],
                        linewidth=CONFIG['estilos']['lineas']['principal']['linewidth'],
                        zorder=1
                    )
                    ax.axhline(
                        params['b'],
                        color=CONFIG['estilos']['lineas']['principal']['color'],
                        linestyle=CONFIG['estilos']['lineas']['principal']['linestyle'],
                        linewidth=CONFIG['estilos']['lineas']['principal']['linewidth'],
                        zorder=1
                    )
                    line_handles.append(
                        Line2D([], [], color=CONFIG['estilos']['lineas']['principal']['color'],
                               linestyle=CONFIG['estilos']['lineas']['principal']['linestyle'],
                               label=params['label'])
                    )
            
            # ===== Configuración final =====
            ax.set(
                xlabel=diagrama['xlabel'],
                ylabel=diagrama['ylabel'],
                title=diagrama['titulo']
            )
            ax.grid(alpha=0.3, linestyle=':')
            
            # ===== Leyenda unificada =====
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            
            final_handles = list(unique.values()) + line_handles
            final_labels = list(unique.keys()) + [h.get_label() for h in line_handles]
            
            ax.legend(
                final_handles,
                final_labels,
                loc='best',
                frameon=True,
                framealpha=0.95,
                fontsize=10,
                handletextpad=0.5,
                borderpad=1
            )
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

def main():
    input_pne = "HASH-IPHAS2arcsec-wise3arcsec.csv"
    input_yso = "YSOs-IPHAS2arcsec-wise3arcsec.csv"
    output_file = "PNe_vs_YSOsHighProb_Analysis.pdf"
    
    try:
        print("🚀 Iniciando análisis...")
        
        # Cargar datos
        print("\n📂 Cargando datos PNe...")
        pne_df = load_data(input_pne, 'PNe')
        print(f"   ➔ PNe cargados: {len(pne_df)} objetos (T: {sum(pne_df.PNstat == 'T')}, "
              f"P: {sum(pne_df.PNstat == 'P')}, L: {sum(pne_df.PNstat == 'L')})")
        
        print("\n📂 Cargando datos YSOs...")
        yso_df = load_data(input_yso, 'YSOs')
        print(f"   ➔ YSOs de alta probabilidad: {len(yso_df)} objetos")
        
        # Calcular colores
        print("\n🎨 Calculando índices de color...")
        pne_colors = calculate_colors(pne_df)
        yso_colors = calculate_colors(yso_df)
        
        # Generar gráficos
        print("\n📊 Generando diagramas...")
        plot_diagrams(pne_df, yso_df, pne_colors, yso_colors, output_file)
        
        print(f"\n✅ Análisis completado exitosamente!\n   Resultados guardados en: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {str(e)}")
        if 'pne_df' in locals():
            print("\nℹ️ Estructura de datos PNe:")
            print(pne_df.info())
        if 'yso_df' in locals():
            print("\nℹ️ Estructura de datos YSOs:")
            print(yso_df.info())
        raise

if __name__ == "__main__":
    main()
