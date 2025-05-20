#!/usr/bin/env python3
"""
Selección de PNEs con manejo correcto de columnas calculadas
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Criterios fotométricos para PNEs
CRITERIOS_PNE = {
    'W1-W2_min': 0.865,
    'J-H_max': 0.75,
    'H-W2_min': 2.7
}

# Configuración visual
ESTILOS = {
    'muestra': {'color': '#95a5a6', 'alpha': 0.3, 's': 20},
    'pne_w1w2': {'color': '#e74c3c', 's': 40, 'label': 'W1-W2 ≥ 0.865', 'marker': '^'},
    'pne_jh': {'color': '#3498db', 's': 40, 'label': 'J-H ≤ 0.75', 'marker': 's'},
    'pne_hw2': {'color': '#2ecc71', 's': 40, 'label': 'H-W2 ≥ 2.7', 'marker': 'o'},
    'final': {'color': '#f1c40f', 's': 60, 'label': 'PNEs únicos', 'edgecolor': 'k'}
}

def cargar_datos(archivo):
    """Carga y valida los datos manteniendo todos los objetos"""
    df = pd.read_csv(archivo)
    print(f"\n[+] Datos cargados: {len(df)} objetos")
    
    # Calcular columnas de color aquí
    df['W1_W2'] = df['W1mag'] - df['W2mag']
    df['J_H'] = df['Jmag'] - df['Hmag']
    df['H_W2'] = df['Hmag'] - df['W2mag']
    
    return df

def aplicar_criterios(df):
    """Aplica los tres criterios de PNE usando las columnas ya calculadas"""
    # Criterio 1: W1-W2 >= 0.865
    pne_w1w2 = df[df['W1_W2'] >= CRITERIOS_PNE['W1-W2_min']].copy()
    pne_w1w2['Criterio'] = 'W1-W2'

    # Criterio 2: J-H <= 0.75
    pne_jh = df[df['J_H'] <= CRITERIOS_PNE['J-H_max']].copy()
    pne_jh['Criterio'] = 'J-H'

    # Criterio 3: H-W2 >= 2.7
    pne_hw2 = df[df['H_W2'] >= CRITERIOS_PNE['H-W2_min']].copy()
    pne_hw2['Criterio'] = 'H-W2'

    return pne_w1w2, pne_jh, pne_hw2

def consolidar_pnes(pne_w1w2, pne_jh, pne_hw2):
    """Combina y elimina duplicados manteniendo el origen del criterio"""
    todos_pnes = pd.concat([pne_w1w2, pne_jh, pne_hw2])
    todos_pnes['Criterios'] = todos_pnes.groupby(todos_pnes.index)['Criterio'].transform(lambda x: '+'.join(x))
    pnes_unicos = todos_pnes[~todos_pnes.index.duplicated(keep='first')].copy()
    return todos_pnes, pnes_unicos

def generar_graficos(df_base, pnes_todos, pnes_unicos, archivo_salida):
    """Genera diagramas usando las columnas calculadas"""
    with PdfPages(archivo_salida) as pdf:
        # Diagrama W1-W2 vs J-H
        fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
        ax.scatter(df_base['W1_W2'], df_base['J_H'], **ESTILOS['muestra'])
        
        for criterio, estilo in zip(['W1-W2', 'J-H', 'H-W2'], 
                                  [ESTILOS['pne_w1w2'], ESTILOS['pne_jh'], ESTILOS['pne_hw2']]):
            mask = pnes_todos['Criterio'] == criterio
            ax.scatter(pnes_todos[mask]['W1_W2'], pnes_todos[mask]['J_H'], **estilo)
        
        ax.scatter(pnes_unicos['W1_W2'], pnes_unicos['J_H'], **ESTILOS['final'])
        ax.set(xlabel='W1 - W2 (mag)', ylabel='J - H (mag)', 
              title='Selección de PNEs por Criterios Individuales')
        ax.legend()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Selección de PNEs por criterios individuales")
    parser.add_argument("archivo", help="Archivo CSV de entrada")
    parser.add_argument("-o", "--output", default="PNe_candidates", help="Nombre base para los archivos de salida")
    parser.add_argument("-g", "--graficos", action="store_true", help="Generar reporte gráfico")
    args = parser.parse_args()

    try:
        # Cargar y procesar datos
        df = cargar_datos(args.archivo)
        pne_w1w2, pne_jh, pne_hw2 = aplicar_criterios(df)
        
        # Guardar resultados
        salida_dir = os.path.abspath("../Resultados_PNE")
        os.makedirs(salida_dir, exist_ok=True)
        
        pne_w1w2.to_csv(f"{salida_dir}/{args.output}_W1W2.csv", index=False)
        pne_jh.to_csv(f"{salida_dir}/{args.output}_JH.csv", index=False)
        pne_hw2.to_csv(f"{salida_dir}/{args.output}_HW2.csv", index=False)
        
        # Consolidar y guardar
        todos_pnes, pnes_unicos = consolidar_pnes(pne_w1w2, pne_jh, pne_hw2)
        pnes_unicos.to_csv(f"{salida_dir}/{args.output}_final.csv", index=False)
        
        print(f"\n[+] Resultados:")
        print(f"- Candidatos W1-W2: {len(pne_w1w2)}")
        print(f"- Candidatos J-H: {len(pne_jh)}")
        print(f"- Candidatos H-W2: {len(pne_hw2)}")
        print(f"- Candidatos únicos totales: {len(pnes_unicos)}")
        print(f"\n[✓] Archivos guardados en: {salida_dir}/")

        # Generar gráficos
        if args.graficos:
            generar_graficos(df, todos_pnes, pnes_unicos, f"{salida_dir}/{args.output}_diagnosticos.pdf")
            print(f"[✓] Reporte gráfico generado: {salida_dir}/{args.output}_diagnosticos.pdf")

    except Exception as e:
        print(f"\n[!] Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
