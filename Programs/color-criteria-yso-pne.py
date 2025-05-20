#!/usr/bin/env python3
"""
Clasificador Avanzado YSO/PNe con Análisis de Fallos Completo
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Criterios fotométricos fundamentales
CRITERIOS = {
    'W1W2_max': 0.865,
    'JH_min': 0.75,
    'HW2_max': 2.7
}

# Configuración visual profesional
ESTILOS = {
    'YSO': {'color': '#e74c3c', 's': 80, 'label': 'YSOs', 'edgecolor': 'k', 'zorder': 5},
    'Falla_W1W2': {'color': '#3498db', 's': 50, 'label': 'Falló W1-W2', 'marker': '^', 'alpha': 0.8},
    'Falla_JH': {'color': '#2ecc71', 's': 50, 'label': 'Falló J-H', 'marker': 's', 'alpha': 0.8},
    'Falla_HW2': {'color': '#f1c40f', 's': 50, 'label': 'Falló H-W2', 'marker': 'o', 'alpha': 0.8},
    'Falla_Multi': {'color': '#9b59b6', 's': 60, 'label': 'Falló Múltiples', 'marker': 'D', 'alpha': 0.7},
    'Falla_Todos': {'color': '#2c3e50', 's': 70, 'label': 'Falló Todos', 'marker': 'X', 'alpha': 0.9},
    'Corte': {'color': '#7f8c8d', 'linestyle': '--', 'linewidth': 1.8}
}

def cargar_datos(archivo):
    """Carga y prepara los datos con validación exhaustiva"""
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"Archivo no encontrado: {archivo}")
    
    df = pd.read_csv(archivo)
    
    # Verificar columnas esenciales
    columnas_requeridas = ['W1mag', 'W2mag', 'Jmag', 'Hmag']
    if not all(col in df.columns for col in columnas_requeridas):
        faltantes = [col for col in columnas_requeridas if col not in df.columns]
        raise ValueError(f"Columnas faltantes: {faltantes}")
    
    # Calcular índices de color
    df['W1W2'] = df['W1mag'] - df['W2mag']
    df['JH'] = df['Jmag'] - df['Hmag']
    df['HW2'] = df['Hmag'] - df['W2mag']
    
    # Identificar datos completos
    df['Datos_Completos'] = df[columnas_requeridas].notnull().all(axis=1)
    
    return df

def clasificar_objetos(df):
    """Clasificación detallada con análisis de fallos combinados"""
    # Máscara para YSOs
    mask_yso = (
        df['W1W2'].le(CRITERIOS['W1W2_max']) &
        df['JH'].ge(CRITERIOS['JH_min']) &
        df['HW2'].le(CRITERIOS['HW2_max']) &
        df['Datos_Completos']
    )
    
    yso = df[mask_yso].copy()
    pne = df[~mask_yso].copy()
    
    # Análisis detallado de PNEs
    pne_completos = pne[pne['Datos_Completos']].copy()
    pne['Razón'] = 'Datos incompletos'
    
    # Máscaras individuales
    falla_w1w2 = ~pne_completos['W1W2'].le(CRITERIOS['W1W2_max'])
    falla_jh = ~pne_completos['JH'].ge(CRITERIOS['JH_min'])
    falla_hw2 = ~pne_completos['HW2'].le(CRITERIOS['HW2_max'])
    
    # Condiciones combinadas
    condiciones = [
        falla_w1w2 & ~falla_jh & ~falla_hw2,
        ~falla_w1w2 & falla_jh & ~falla_hw2,
        ~falla_w1w2 & ~falla_jh & falla_hw2,
        falla_w1w2 & falla_jh & ~falla_hw2,
        falla_w1w2 & ~falla_jh & falla_hw2,
        ~falla_w1w2 & falla_jh & falla_hw2,
        falla_w1w2 & falla_jh & falla_hw2
    ]
    
    resultados = [
        'Solo W1W2',
        'Solo JH',
        'Solo HW2',
        'W1W2 + JH',
        'W1W2 + HW2',
        'JH + HW2',
        'Todos los criterios'
    ]
    
    pne_completos['Razón'] = np.select(condiciones, resultados, default='Múltiples')
    pne.loc[pne_completos.index, 'Razón'] = pne_completos['Razón']
    
    return yso, pne

def generar_graficos(yso, pne, archivo_salida):
    """Genera diagramas científicos de alta calidad"""
    with PdfPages(archivo_salida) as pdf:
        # Configuración común
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
        
        # --- Diagrama 1: W1-W2 vs J-H ---
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        # Capas de datos
        capas = [
            ('Todos los criterios', ESTILOS['Falla_Todos']),
            ('W1W2 + JH', ESTILOS['Falla_Multi']),
            ('W1W2 + HW2', ESTILOS['Falla_Multi']),
            ('JH + HW2', ESTILOS['Falla_Multi']),
            ('Solo W1W2', ESTILOS['Falla_W1W2']),
            ('Solo JH', ESTILOS['Falla_JH']),
            ('Solo HW2', ESTILOS['Falla_HW2'])
        ]
        
        pne_completos = pne[pne['Datos_Completos']]
        for razón, estilo in capas:
            mask = pne_completos['Razón'] == razón
            ax.scatter(
                pne_completos[mask]['W1W2'],
                pne_completos[mask]['JH'],
                **estilo
            )
        
        # YSOs
        ax.scatter(yso['W1W2'], yso['JH'], **ESTILOS['YSO'])
        
        # Líneas de corte
        ax.axvline(CRITERIOS['W1W2_max'], **ESTILOS['Corte'], 
                  label=f'W1-W2 ≤ {CRITERIOS["W1W2_max"]}')
        ax.axhline(CRITERIOS['JH_min'], **ESTILOS['Corte'],
                  label=f'J-H ≥ {CRITERIOS["JH_min"]}')
        
        # Anotaciones profesionales
        ax.text(0.97, 0.95, 
                f"Criterio Adicional:\nH-W2 ≤ {CRITERIOS['HW2_max']}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='#34495e', boxstyle='round'),
                fontsize=10, color='#2c3e50')
        
        ax.set(xlabel=r'$W1 - W2$ (mag)', ylabel=r'$J - H$ (mag)',
              title='Clasificación YSO/PNe: W1-W2 vs J-H\n(Criterio H-W2 no visible)')
        ax.legend(loc='lower left', frameon=True, framealpha=0.95)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # --- Diagrama 2: H-W2 vs J-H ---
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        for razón, estilo in capas:
            mask = pne_completos['Razón'] == razón
            ax.scatter(
                pne_completos[mask]['HW2'],
                pne_completos[mask]['JH'],
                **estilo
            )
        
        ax.scatter(yso['HW2'], yso['JH'], **ESTILOS['YSO'])
        
        ax.axvline(CRITERIOS['HW2_max'], **ESTILOS['Corte'],
                  label=f'H-W2 ≤ {CRITERIOS["HW2_max"]}')
        ax.axhline(CRITERIOS['JH_min'], **ESTILOS['Corte'])
        
        ax.text(0.97, 0.95, 
                f"Criterio Adicional:\nW1-W2 ≤ {CRITERIOS['W1W2_max']}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='#34495e', boxstyle='round'),
                fontsize=10, color='#2c3e50')
        
        ax.set(xlabel=r'$H - W2$ (mag)', ylabel=r'$J - H$ (mag)',
              title='Clasificación YSO/PNe: H-W2 vs J-H\n(Criterio W1-W2 no visible)')
        ax.legend(loc='lower left', frameon=True, framealpha=0.95)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Clasificador Avanzado de Objetos Estelares")
    parser.add_argument("input", help="Archivo CSV de entrada")
    parser.add_argument("-o", "--output", default="Clasificacion", 
                       help="Prefijo para nombres de archivos de salida")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Modo detallado de salida")
    args = parser.parse_args()
    
    try:
        # Procesamiento inicial
        if args.verbose:
            print("\n[+] Iniciando procesamiento de datos...")
            print(f" - Archivo de entrada: {args.input}")
        
        df = cargar_datos(args.input)
        yso, pne = clasificar_objetos(df)
        
        # Configurar salida
        salida_dir = os.path.abspath("../Resultados_Avanzados")
        os.makedirs(salida_dir, exist_ok=True)
        base_salida = os.path.join(salida_dir, args.output)
        
        # Guardar resultados detallados
        if args.verbose:
            print("[+] Guardando resultados...")
        
        # YSOs
        yso.to_csv(f"{base_salida}_YSOs.csv", index=False)
        
        # PNEs completos con diferentes categorías
        pne_completos = pne[pne['Datos_Completos']]
        for categoria in pne_completos['Razón'].unique():
            mask = pne_completos['Razón'] == categoria
            nombre_cat = categoria.replace(" ", "_").replace("+", "_").lower()
            pne_completos[mask].to_csv(f"{base_salida}_PNes_{nombre_cat}.csv", index=False)
        
        # PNEs con datos incompletos
        pne_incompletos = pne[~pne['Datos_Completos']]
        pne_incompletos.to_csv(f"{base_salida}_PNes_datos_incompletos.csv", index=False)
        
        # Generar reporte gráfico
        if args.verbose:
            print("[+] Generando reporte visual...")
        
        generar_graficos(yso, pne, f"{base_salida}_Diagnosticos.pdf")
        
        # Reporte final
        if args.verbose:
            print("\n[✓] Proceso completado exitosamente!")
            print("---------------------------------------")
            print(f"Total objetos procesados: {len(df)}")
            print(f"YSOs identificados: {len(yso)}")
            print(f"PNEs clasificados: {len(pne)}")
            print(f" - Por categoría:")
            for cat, count in pne_completos['Razón'].value_counts().items():
                print(f"   - {cat}: {count}")
            print(f" - Datos incompletos: {len(pne_incompletos)}")
            print(f"\nArchivos guardados en: {salida_dir}/")
            print(f"Reporte gráfico: {base_salida}_Diagnosticos.pdf")
        
    except Exception as e:
        print(f"\n[!] Error crítico: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
