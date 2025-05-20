#!/usr/bin/env python3
"""
Remove rows from df1 that have a matching 'Name' in df2.
"""

import pandas as pd

def validate_data(df):
    """Valida y limpia los datos manteniendo todas las columnas"""
    required_cols = ["W1mag", "W2mag", "Jmag", "Hmag"]
    
    # Verificar existencia de columnas requeridas
    if missing := list(set(required_cols) - set(df.columns)):
        raise ValueError(f"Columnas faltantes: {missing}")
    
    # Filtrar filas con valores nulos en columnas requeridas
    df_clean = df.dropna(subset=required_cols).copy()
    
    if df_clean.empty:
        raise ValueError("Datos insuficientes después de limpieza")
    
    return df_clean

# Cargar y validar df1
df1 = pd.read_csv("../Class_wise_v4/Halpha_emitter_wise_group4.csv")
df1 = validate_data(df1)

# Cargar df2 y validar 'Name'
df2 = pd.read_csv("YSOs_Halpha_emitter_wise_group4.csv")
if 'Name' not in df2.columns:
    raise ValueError("La columna 'Name' no existe en df2")
if df2['Name'].isnull().any():
    raise ValueError("La columna 'Name' en df2 tiene valores nulos")

# Normalizar nombres
df1['Name'] = df1['Name'].str.strip().str.lower()
df2['Name'] = df2['Name'].str.strip().str.lower()

# Eliminar duplicados en df2 (opcional)
df2 = df2.drop_duplicates(subset=['Name'])

# Filtrar df1
print(f"Filas originales en df1: {len(df1)}")
df_final = df1[~df1['Name'].isin(df2['Name'])]
print(f"Filas después de filtrar: {len(df_final)}")

# Guardar el resultado
df_final.to_csv("PNe_candidates.csv", index=False)
