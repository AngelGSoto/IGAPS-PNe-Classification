#!/usr/bin/env python3
"""
Script mejorado para selección de candidatos a nebulosas planetarias
Incluye validaciones adicionales y opción para unión/intersección de criterios
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Configuración de parámetros científicos
CRITERIA_PARAMS = {
    'criterion1': {'a': -0.5392, 'b': 1.8268},
    'criterion2': {'a': -0.1473, 'b': 1.9123},
    'criterion3': {'a': 1.6563, 'b': 0.7335},
    'criterion4': {'a': 0.1579, 'b': 2.2092},
    'criterion5': {'a': 1.1100, 'b': 2.2339},
    'criterion6': {'a': 2.5943, 'b': 0.8026},
    'criterion7': {'a': 1.1098, 'b': 0.8292}
}

def objective(x, a, b):
    """Función lineal para criterios de selección"""
    return a * x + b

def calculate_colors(df):
    """Calcula todos los colores necesarios"""
    return {
        'c1_x': df["W1mag"] - df["W2mag"],
        'c1_y': df["rImag"] - df["Hamag"],
        'c2_x': df["gmag"] - df["rImag"],
        'c2_y': df["Hmag"] - df["W1mag"],
        'c3_x': df["Jmag"] - df["Hmag"],
        'c3_y': df["Hmag"] - df["W2mag"],
        'c4_x': df["Jmag"] - df["Hmag"],
        'c4_y': df["Jmag"] - df["W2mag"],
        'c5_x': df["Jmag"] - df["Kmag"],
        'c5_y': df["Jmag"] - df["W1mag"]
    }

def validate_data(df):
    """Valida estructura y calidad de los datos"""
    required_cols = ["W1mag", "W2mag", "rImag", "Hamag", "gmag", "Hmag", "Jmag", "Kmag"]
    if missing := list(set(required_cols) - set(df.columns)):
        raise ValueError(f"Columnas faltantes: {missing}")
    
    if df[required_cols].isnull().all().any():
        raise ValueError("Datos insuficientes en columnas críticas")
        
    return df.dropna(subset=required_cols).copy()

def generate_plots(df, colors, masks, output_path):
    """Genera diagramas de color-color en PDF"""
    with PdfPages(output_path) as pdf:
        # Diagrama para Criterio 1
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(colors['c1_x'], colors['c1_y'], c='gray', alpha=0.5, label='Muestra completa')
        ax.scatter(colors['c1_x'][masks['combined']], colors['c1_y'][masks['combined']], 
                  c='red', s=50, label='Candidatos')
        x_vals = np.linspace(min(colors['c1_x']), max(colors['c1_x']), 100)
        ax.plot(x_vals, objective(x_vals, **CRITERIA_PARAMS['criterion1']), 
               'k--', lw=1.5, label='Límite criterio 1')
        ax.set(xlabel="W1 - W2 (mag)", ylabel="rI - Ha (mag)",
              title="Diagrama de Color-Color: Criterio 1")
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        # Diagrama para Criterio 2
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(colors['c2_x'], colors['c2_y'], c='gray', alpha=0.5, label='Muestra completa')
        ax.scatter(colors['c2_x'][masks['combined']], colors['c2_y'][masks['combined']], 
                  c='blue', s=50, label='Candidatos')
        x_vals = np.linspace(min(colors['c2_x']), max(colors['c2_x']), 100)
        ax.plot(x_vals, objective(x_vals, **CRITERIA_PARAMS['criterion2']), 
               'k--', lw=1.5, label='Límite criterio 2')
        ax.set(xlabel="g - rI (mag)", ylabel="H - W1 (mag)",
              title="Diagrama de Color-Color: Criterio 2")
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        # Diagrama para Criterio 3
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(colors['c3_x'], colors['c3_y'], c='gray', alpha=0.5, label='Muestra completa')
        ax.scatter(colors['c3_x'][masks['combined']], colors['c3_y'][masks['combined']], 
                  c='blue', s=50, label='Candidatos')
        x_vals = np.linspace(min(colors['c3_x']), max(colors['c3_x']), 100)
        ax.plot(x_vals, objective(x_vals, **CRITERIA_PARAMS['criterion3']),
                'k--', lw=1.5, label='Límite criterio 3')
        ax.plot(x_vals, objective(x_vals, **CRITERIA_PARAMS['criterion4']),
                'k--', lw=1.5)
        ax.set(xlabel="J - H (mag)", ylabel="H - W2 (mag)",
              title="Diagrama de Color-Color: Criterio 3")
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        # Diagrama para Criterio 4
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(colors['c4_x'], colors['c4_y'], c='gray', alpha=0.5, label='Muestra completa')
        ax.scatter(colors['c4_x'][masks['combined']], colors['c4_y'][masks['combined']], 
                  c='blue', s=50, label='Candidatos')
        x_vals = np.linspace(min(colors['c4_x']), max(colors['c4_x']), 100)
        ax.plot(x_vals, objective(x_vals, **CRITERIA_PARAMS['criterion5']),
                'k--', lw=1.5, label='Límite criterio 4')
        ax.plot(x_vals, objective(x_vals, **CRITERIA_PARAMS['criterion6']),
                'k--', lw=1.5)
        ax.set(xlabel="J - H (mag)", ylabel="J - W2 (mag)",
              title="Diagrama de Color-Color: Criterio 4")
        ax.legend()
        pdf.savefig(fig)
        plt.close()

         # Diagrama para Criterio 5
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(colors['c5_x'], colors['c5_y'], c='gray', alpha=0.5, label='Muestra completa')
        ax.scatter(colors['c5_x'][masks['combined']], colors['c5_y'][masks['combined']], 
                  c='blue', s=50, label='Candidatos')
        x_vals = np.linspace(min(colors['c5_x']), max(colors['c5_x']), 100)
        ax.plot(x_vals, objective(x_vals, **CRITERIA_PARAMS['criterion7']),
                'k--', lw=1.5, label='Límite criterio 5')
        ax.set(xlabel="J - K (mag)", ylabel="J - W1 (mag)",
              title="Diagrama de Color-Color: Criterio 5")
        ax.legend()
        pdf.savefig(fig)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Selección de candidatos a PNe")
    parser.add_argument("input_base", help="Base del nombre del archivo (sin extensión)")
    parser.add_argument("--logic", choices=['and', 'or'], default='and',
                       help="Lógica para combinar criterios (and/or)")
    parser.add_argument("--plot", action="store_true", help="Generar diagramas PDF")
    parser.add_argument("--verbose", action="store_true", help="Modo verboso")
    args = parser.parse_args()

    # Configuración de archivos
    input_file = f"{args.input_base}.csv"
    output_dir = "../PN_candidates"
    os.makedirs(output_dir, exist_ok=True)
    
    output_base = os.path.join(output_dir, f"PNe-{os.path.basename(args.input_base)}")
    
    try:
        # Carga y validación de datos
        if args.verbose:
            print(f"[+] Cargando datos: {input_file}")
            
        df = pd.read_csv(input_file)
        clean_df = validate_data(df)
        
        if args.verbose:
            print(f"[+] Datos limpios: {len(clean_df)}/{len(df)} registros")
        
        # Cálculo de colores
        colors = calculate_colors(clean_df)
        
        # Aplicación de criterios
        criterion1 = colors['c1_y'] >= objective(colors['c1_x'], **CRITERIA_PARAMS['criterion1'])
        criterion2 = colors['c2_y'] >= objective(colors['c2_x'], **CRITERIA_PARAMS['criterion2'])
        criterion3 = colors['c3_y'] >= objective(colors['c3_x'], **CRITERIA_PARAMS['criterion3'])
        criterion4 = colors['c3_y'] >= objective(colors['c3_x'], **CRITERIA_PARAMS['criterion4'])
        criterion5 = colors['c4_y'] >= objective(colors['c4_x'], **CRITERIA_PARAMS['criterion5'])
        criterion6 = colors['c4_y'] >= objective(colors['c4_x'], **CRITERIA_PARAMS['criterion6'])
        criterion7 = colors['c5_y'] >= objective(colors['c5_x'], **CRITERIA_PARAMS['criterion7'])
        
        # Combinación de criterios
        combine_logic = {
            'and': criterion1 & criterion2 & criterion3 & criterion4 & criterion5 & criterion6 & criterion7,
            'or': criterion1 | criterion2 | criterion3 | criterion4 | criterion5 | criterion6 | criterion7
        }
        final_mask = combine_logic[args.logic]
        
        # Selección final
        candidates = clean_df[final_mask].reset_index(drop=True)
        
        if args.verbose:
            print(f"\n[+] Resultados de selección:")
            print(f" - Criterio 1: {criterion1.sum()} candidatos")
            print(f" - Criterio 2: {criterion2.sum()} candidatos")
            print(f" - Criterio 3: {criterion3.sum()} candidatos")
            print(f" - Criterio 4: {criterion4.sum()} candidatos")
            print(f" - Criterio 5: {criterion5.sum()} candidatos")
            print(f" - Criterio 6: {criterion6.sum()} candidatos")
            print(f" - Criterio 7: {criterion7.sum()} candidatos")
            print(f" - Combinación ({args.logic.upper()}): {len(candidates)} candidatos\n")
        
        # Guardado de resultados
        output_csv = f"{output_base}-candidates-stricted.csv"
        candidates.to_csv(output_csv, index=False)
        
        if args.verbose:
            print(f"[+] Candidatos guardados en: {output_csv}")
        
        # Generación de gráficos
        if args.plot:
            output_pdf = f"{output_base}-diagrams-stricted.pdf"
            generate_plots(clean_df, colors, {'combined': final_mask}, output_pdf)
            
            if args.verbose:
                print(f"[+] Diagramas guardados en: {output_pdf}")
                
    except Exception as e:
        print(f"\n[!] Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
