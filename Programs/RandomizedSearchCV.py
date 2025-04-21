from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap

# 1. Optimización de memoria
def reduce_mem_usage(df):
    """Reduce el uso de memoria convirtiendo a tipos más eficientes"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # float16 puede dar problemas
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.read_csv("IGAPs-emitters-wise.csv")

# Máximos errores permitidos (ajustables según necesidades)
max_err_optical = 0.2  # Para bandas ópticas (g, r, i, Hα, U)
max_err_wise = 0.5      # Para WISE (W1, W2)
max_err_2mass = 0.2     # Para 2MASS (J, H, K)

# Máscara de errores para IGAPS (óptico)
m_err_igaps = (
    (combined_df["e_imag"] <= max_err_optical) &
    (combined_df["e_Hamag"] <= max_err_optical) &
    (combined_df["e_rImag"] <= max_err_optical) &
    (combined_df["e_rUmag"] <= max_err_optical) &
    (combined_df["e_gmag"] <= max_err_optical) &
    (combined_df["e_Umag"] <= max_err_optical)  # Opcional si se usa UV
)

# Filtrado de flags de calidad (ej. eliminar saturados)
m_quality_igaps = (
    (combined_df["Saturatedi"] == 0) &
    (combined_df["BadPixi"] == 0) &
    (combined_df["Traili"] == 0)  # Aplica a todas las bandas relevantes
)

# Máscara de errores para WISE y 2MASS
m_err_wise_tmass = (
    (combined_df["e_W1mag"] <= max_err_wise) &
    (combined_df["e_W2mag"] <= max_err_wise) &
    (combined_df["e_Jmag"] <= max_err_2mass) &
    (combined_df["e_Hmag"] <= max_err_2mass) &
    (combined_df["e_Kmag"] <= max_err_2mass)
)

# Combinar todas las máscaras
mask_total = (
    m_err_igaps &
    m_quality_igaps &
    m_err_wise_tmass 
)

# Aplicar la máscara al DataFrame
df_filtered = combined_df[mask_total].copy()

#Selecting columns
columns = ["rImag",       # Magnitud en banda r (IPHAS)
    "Hamag",       # Magnitud en Hα (filtro estrecho)
    "gmag",        # Magnitud en banda g
    "imag",        # Magnitud en banda i
    "rUmag",       # Magnitud en banda r (UVEX, para variabilidad)
    "Umag",
    "W1mag",       # WISE 3.4 µm
    "W2mag",       # WISE 4.6 µm
    "Jmag",        # 2MASS J (1.25 µm)
    "Hmag",        # 2MASS H (1.65 µm)
    "Kmag"         # 2MASS Ks (2.17 µm)]
          ]

df_mag = df_filtered[columns]

# Creación de todos los colores no redundantes
df_colors = df_mag.assign(
    # --- Colores Óptico-Óptico (IGAPS) ---
    U_g = lambda x: x["Umag"] - x["gmag"],
    g_r = lambda x: x["gmag"] - x["rImag"],
    r_i = lambda x: x["rImag"] - x["imag"],
    r_Ha = lambda x: x["rImag"] - x["Hamag"],   # Exceso de Hα principal
    i_Ha = lambda x: x["imag"] - x["Hamag"],     # Alternativo para objetos rojos
    
    # --- Colores Óptico-IR (IGAPS + WISE/2MASS) ---
    g_W1 = lambda x: x["gmag"] - x["W1mag"],     # Exceso IR en objetos azules
    r_W2 = lambda x: x["rImag"] - x["W2mag"],    # Emisión óptica vs polvo térmico
    Ha_W1 = lambda x: x["Hamag"] - x["W1mag"],   # Hα vs polvo cálido
    Ha_W2 = lambda x: x["Hamag"] - x["W2mag"],   # Hα vs polvo más frío
    i_K = lambda x: x["imag"] - x["Kmag"],       # Óptico rojo vs IR cercano
    J_r = lambda x: x["Jmag"] - x["rImag"],      # IR vs óptico
    
    # --- Colores IR-IR (WISE + 2MASS) ---
    W1_W2 = lambda x: x["W1mag"] - x["W2mag"],   # Exceso térmico clave
    J_H = lambda x: x["Jmag"] - x["Hmag"],       # Indicador de tipo espectral
    H_K = lambda x: x["Hmag"] - x["Kmag"],       # Exceso en K (polvo/CO)
    W1_J = lambda x: x["W1mag"] - x["Jmag"],     # Polvo vs componente estelar
    
    # --- Variabilidad ---
    var_r = lambda x: np.abs(x["rImag"] - x["rUmag"])  # Absoluta para evitar negativos
)

# Lista final de features para UMAP/HDBSCAN
features = [
    # Óptico-Óptico
    'U_g', 'g_r', 'r_i', 'r_Ha', 'i_Ha',
    
    # Óptico-IR
    'g_W1', 'r_W2', 'Ha_W1', 'Ha_W2', 'i_K', 'J_r',
    
    # IR-IR
    'W1_W2', 'J_H', 'H_K', 'W1_J',
    
    # Variabilidad
    'var_r'
]

# Filtrar NaNs y crear dataset final
df_analysis = df_colors[features].dropna()
# Aplicar reducción de memoria a tus datos
df_analysis = reduce_mem_usage(df_analysis)

# Verificación rápida
print(f"Columnas finales: {df_analysis.columns.tolist()}")
print(f"Tamaño del dataset: {df_analysis.shape}")

#Pipeline

# 2. Muestreo aleatorio para la optimización (ajustar el tamaño según tu RAM)
sample_size = min(10000, len(df_analysis))  # Empieza con 10,000 observaciones
df_sample = df_analysis.sample(sample_size, random_state=42)

X_stand = StandardScaler().fit_transform(df_sample)

# 3. Pipeline optimizado
pipeline = make_pipeline(
    umap.UMAP(random_state=42, verbose=False),  # Desactivar mensajes
    MiniBatchKMeans(random_state=42, n_init='auto')  # KMeans más rápido
)

# 4. Espacio de parámetros reducido y optimizado
param_dist = {
    'umap__n_components': [2, 3],  # Empezar con dimensiones bajas
    'umap__n_neighbors': [15, 20], # Reducir opciones
    'umap__min_dist': [0.1],
    'umap__n_epochs': [50],        # Reducir epochs para mayor velocidad
    'minibatchkmeans__n_clusters': [3, 4]  # Menos opciones de clusters
}

# 5. Búsqueda optimizada
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=5,  # Empezar con pocas iteraciones
    scoring='silhouette',
    cv=3,      # Reducir validación cruzada
    verbose=1,
    n_jobs=1   # Evitar paralelización si hay poca RAM
)

# 6. Entrenamiento con monitorización de memoria
try:
    search.fit(X_stand)
except MemoryError:
    print("¡Error de memoria! Reduce el sample_size o los parámetros")

# 7. Visualización ligera
if search.cv_results_:
    results_df = pd.DataFrame(search.cv_results_)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results_df,
        x='param_umap__n_neighbors',
        y='mean_test_score',
        hue='param_umap__n_components',
        size='param_minibatchkmeans__n_clusters',
        sizes=(20, 200)
    )
    plt.title('Optimización de Parámetros')
    plt.show()
else:
    print("No se completó ninguna iteración debido a restricciones de memoria")
