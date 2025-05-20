#!/usr/bin/env python3
"""
Selección de candidatos PNe/YSO con separación absoluta del archivo original
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Criterios fotométricos
CRITERIA = {
    'W1-W2_max': 0.865,
    'J-H_min': 0.75,
    'H-W2_max': 2.7
}

# Configuración visual
PLOT_STYLE = {
    'full_sample': {'color': '#95a5a6', 'alpha': 0.3, 's': 20, 'label': 'Muestra completa'},
    'yso': {'color': '#e74c3c', 's': 50, 'label': 'YSO candidates'},
    'pne_clean': {'color': '#3498db', 's': 50, 'label': 'PNe (datos completos)'},
    'pne_missing': {'color': '#f1c40f', 's': 30, 'label': 'Datos faltantes', 'alpha': 0.5},
    'cut_lines': {'color': '#2c3e50', 'linestyle': '--', 'linewidth': 1.5}
}

def calcular_colores(df):
    """Calcula índices de color con validación de bandas"""
    bandas_requeridas = ['W1mag', 'W2mag', 'Jmag', 'Hmag']
    if faltantes := list(set(bandas_requeridas) - set(df.columns)):
        raise ValueError(f"Bandas faltantes en datos: {faltantes}")
    
    return pd.DataFrame({
        'W1-W2': df['W1mag'] - df['W2mag'],
        'J-H': df['Jmag'] - df['Hmag'],
        'H-W2': df['Hmag'] - df['W2mag']
    })

def procesar_datos(archivo_entrada, logica, generar_graficos_flag, verbose):
    """Flujo principal de procesamiento de datos"""
    # Cargar datos crudos
    datos_crudos = pd.read_csv(f"{archivo_entrada}.csv")
    if verbose:
        print(f"\n[+] Datos originales cargados: {len(datos_crudos)} objetos")

    # Limpiar datos (solo para selección YSO)
    datos_limpios = datos_crudos.dropna(subset=['W1mag', 'W2mag', 'Jmag', 'Hmag']).copy()
    if datos_limpios.empty:
        raise ValueError("No hay datos válidos para procesar criterios YSO")
    
    if verbose:
        print(f"  - Datos limpios (sin valores faltantes): {len(datos_limpios)} objetos")

    # Calcular colores y aplicar criterios
    colores = calcular_colores(datos_limpios)
    mascaras = {
        'c1': colores['W1-W2'] <= CRITERIA['W1-W2_max'],
        'c2': colores['J-H'] >= CRITERIA['J-H_min'],
        'c3': colores['H-W2'] <= CRITERIA['H-W2_max']
    }

    # Combinar criterios
    if logica == 'y':
        mascara_yso = mascaras['c1'] & mascaras['c2'] & mascaras['c3']
    else:
        mascara_yso = mascaras['c1'] | mascaras['c2'] | mascaras['c3']

    yso_candidates = datos_limpios[mascara_yso]
    pne_candidates = datos_crudos[~datos_crudos.index.isin(yso_candidates.index)]

    # Guardar resultados
    directorio_salida = os.path.abspath("../Resultados_PNe_YSO_finales")
    os.makedirs(directorio_salida, exist_ok=True)

    nombre_base = os.path.basename(archivo_entrada)
    yso_candidates.to_csv(f"{directorio_salida}/YSOs_{nombre_base}.csv", index=False)
    pne_candidates.to_csv(f"{directorio_salida}/PNes_{nombre_base}.csv", index=False)

    if verbose:
        print(f"\n[+] Resultados finales:")
        print(f"YSOs identificados: {len(yso_candidates)}")
        print(f"PNes candidatos: {len(pne_candidates)}")
        print(f"  - Incluye {len(datos_crudos)-len(datos_limpios)} objetos con datos faltantes")
        print(f"\n[✓] Archivos guardados en:")
        print(f"  - YSOs: {directorio_salida}/YSOs_{nombre_base}.csv")
        print(f"  - PNes: {directorio_salida}/PNes_{nombre_base}.csv")

    # Generar gráficos solo con datos limpios
    if generar_graficos_flag:
        with PdfPages(f"{directorio_salida}/Diagnosticos_{nombre_base}.pdf") as pdf:
            fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
            
            # Diagrama W1-W2 vs J-H
            ax.scatter(colores['W1-W2'], colores['J-H'], **PLOT_STYLE['full_sample'])
            ax.scatter(colores['W1-W2'][mascara_yso], colores['J-H'][mascara_yso], **PLOT_STYLE['yso'])
            ax.scatter(colores['W1-W2'][~mascara_yso], colores['J-H'][~mascara_yso], **PLOT_STYLE['pne_clean'])
            
            ax.axvline(CRITERIA['W1-W2_max'], **PLOT_STYLE['cut_lines'], label=f"W1-W2 ≤ {CRITERIA['W1-W2_max']}")
            ax.axhline(CRITERIA['J-H_min'], **PLOT_STYLE['cut_lines'], label=f"J-H ≥ {CRITERIA['J-H_min']}")
            
            ax.set(xlabel='W1 - W2 (mag)', ylabel='J - H (mag)', 
                  title='Diagrama de Selección: YSO vs PNe (datos completos)')
            ax.legend()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selección estricta de YSOs y PNe del archivo original")
    parser.add_argument("archivo_base", help="Nombre base del archivo (sin extensión)")
    parser.add_argument("-l", "--logica", choices=['y', 'o'], default='y',
                      help="Lógica de combinación: 'y' (AND) u 'o' (OR)")
    parser.add_argument("-p", "--plot", action="store_true", help="Generar reporte gráfico")
    parser.add_argument("-v", "--verbose", action="store_true", help="Modo detallado")
    
    args = parser.parse_args()
    
    try:
        procesar_datos(
            archivo_entrada=args.archivo_base,
            logica=args.logica,
            generar_graficos_flag=args.plot,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\n[!] Error crítico: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)
