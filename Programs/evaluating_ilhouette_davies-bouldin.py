import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import pandas as pd
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from itertools import product
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Función para calcular métricas de validación
def calcular_metricas(reductor, kmeans, X_entrenamiento, X_validacion):
    # Transformar datos
    X_entrenamiento_trans = reductor.transform(X_entrenamiento)
    X_validacion_trans = reductor.transform(X_validacion)
    
    # Predecir clusters
    etiquetas_entrenamiento = kmeans.predict(X_entrenamiento_trans)
    etiquetas_validacion = kmeans.predict(X_validacion_trans)
    
    # Calcular métricas
    metricas = {
        'silueta_entrenamiento': silhouette_score(X_entrenamiento_trans, etiquetas_entrenamiento),
        'silueta_validacion': silhouette_score(X_validacion_trans, etiquetas_validacion),
        'db_entrenamiento': davies_bouldin_score(X_entrenamiento_trans, etiquetas_entrenamiento),
        'db_validacion': davies_bouldin_score(X_validacion_trans, etiquetas_validacion)
    }
    return metricas

# Cargar y preparar datos
df = pd.read_csv("IGAPs-emitters.csv")

# Filtrar fuentes
mascara = (
    (df['SaturatedHa'] == 0) &
    (df['DeblendHa'] == 0) &
    (df['BadPixHa'] == 0) &
    (df['errBits'] == 0))
df_filtrado = df[mascara].copy()

# Calcular variables
df_filtrado['r_Ha'] = df_filtrado['rImag'] - df_filtrado['Hamag']
df_filtrado['r_i'] = df_filtrado['rImag'] - df_filtrado['imag']
df_filtrado['g_r'] = df_filtrado['gmag'] - df_filtrado['rImag']
df_filtrado['U_g'] = df_filtrado['Umag'] - df_filtrado['gmag']
df_filtrado['Ha_i'] = df_filtrado['Hamag'] - df_filtrado['imag']
df_filtrado['var_r'] = df_filtrado['rImag'] - df_filtrado['rUmag']

variables = ['r_Ha', 'r_i', 'g_r', 'U_g', 'Ha_i', 'var_r']
df_limpio = df_filtrado[variables].dropna()

# Estandarizar datos
X_estandarizado = StandardScaler().fit_transform(df_limpio[variables].values)

# Dividir en entrenamiento y validación
X_entrenamiento, X_validacion = train_test_split(X_estandarizado, test_size=0.2, random_state=42)

# Rangos de parámetros a explorar
componentes_range = [2, 3, 4, 5]
vecinos_range = [5, 15, 30, 50]

# Función paralelizable para búsqueda de parámetros
def buscar_parametros(params):
    n_componentes, n_vecinos = params
    
    # Entrenar UMAP
    reductor = umap.UMAP(
        n_components=n_componentes,
        n_neighbors=n_vecinos,
        random_state=42
    )
    X_entrenamiento_trans = reductor.fit_transform(X_entrenamiento)
    
    # Entrenar KMeans (clusters = componentes para comparación justa)
    kmeans = KMeans(n_clusters=n_componentes, random_state=42)
    kmeans.fit(X_entrenamiento_trans)
    
    # Calcular métricas
    metricas = calcular_metricas(reductor, kmeans, X_entrenamiento, X_validacion)
    
    return {
        'Componentes': n_componentes,
        'Vecinos': n_vecinos,
        **metricas
    }

# Ejecutar búsqueda en paralelo
parametros = list(product(componentes_range, vecinos_range))
resultados = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(buscar_parametros)(param) for param in parametros
)

# Convertir resultados a DataFrame
df_resultados = pd.DataFrame(resultados)

# Seleccionar mejor modelo (compromiso entre métricas de validación)
df_resultados['puntaje_compuesto'] = (
    df_resultados['silueta_validacion'] - 
    df_resultados['db_validacion']/df_resultados['db_validacion'].max()
)

mejor_modelo = df_resultados.loc[df_resultados['puntaje_compuesto'].idxmax()]

print("\nMejores parámetros encontrados:")
print(f"Componentes UMAP: {mejor_modelo['Componentes']}")
print(f"Vecinos UMAP: {mejor_modelo['Vecinos']}")
print(f"Silueta (validación): {mejor_modelo['silueta_validacion']:.3f}")
print(f"Davies-Bouldin (validación): {mejor_modelo['db_validacion']:.3f}")

# Visualización de resultados
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid", palette="muted")

# Gráfico de Silueta
plt.subplot(1, 2, 1)
sns.lineplot(
    x='Vecinos', 
    y='silueta_validacion', 
    hue='Componentes',
    data=df_resultados,
    marker='o',
    palette='viridis',
    legend='full'
)
plt.title('Puntaje Silueta en Validación')
plt.ylim(0.33, 0.5)

# Gráfico de Davies-Bouldin
plt.subplot(1, 2, 2)
sns.lineplot(
    x='Vecinos', 
    y='db_validacion', 
    hue='Componentes',
    data=df_resultados,
    marker='o',
    palette='viridis',
    legend=False
)
plt.title('Índice Davies-Bouldin en Validación')
plt.ylim(0.75, 1.2)

plt.tight_layout()
plt.savefig("Figs/Optimizacion_UMAP_IGAPS.png", dpi=300)
plt.show()

# Entrenar modelo final con mejores parámetros
mejor_reductor = umap.UMAP(
    n_components=int(mejor_modelo['Componentes']),
    n_neighbors=int(mejor_modelo['Vecinos']),
    random_state=42
).fit(X_estandarizado)

# Guardar modelo para uso futuro
joblib.dump(mejor_reductor, 'mejor_modelo_umap.pkl')
