import pandas as pd
import numpy as np
from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rdtg-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARPETA = Path(__file__).resolve().parent
RESULTADOS = CARPETA / "resultados"
GRAFICOS = CARPETA / "graficos"
RESULTADOS.mkdir(exist_ok=True)
GRAFICOS.mkdir(exist_ok=True)

# Cargamos los datos del dataset original
df_original = pd.read_csv(CARPETA / 'TMBD Movie Dataset.csv')
print(f"Dimensiones iniciales: {df_original.shape[0]} filas, {df_original.shape[1]} columnas")

df = df_original.copy()

# Eliminación de películas duplicadas por ID
filas_duplicadas = df.duplicated(subset=['id']).sum()
if filas_duplicadas > 0:
    df = df.drop_duplicates(subset=['id'], keep='first')

# Filtrado inicial por umbral comercial de presupuesto y recaudación
len_antes_br = len(df)
budget_revenue = (df['budget'] > 10000) & (df['revenue'] > 10000)
df_clean = df[budget_revenue].copy()
filas_eliminadas_br = len_antes_br - len(df_clean)
print(f"Se eliminaron {filas_eliminadas_br} filas con presupuesto o recaudación incoherentes")


# Convertir fechas a formato datetime
df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce')
# El año separado se ha contrastado con la fuente pública. Corregimos 12 siglos.
fechas_incoherentes = (df_clean['release_date'].dt.year != df_clean['release_year']) & df_clean['release_date'].notna()
fechas = df_clean.loc[fechas_incoherentes, ['id', 'original_title', 'release_date']].copy()
fechas.columns = ['id', 'original_title', 'fecha_recibida']
fechas_nuevas = (df_clean.loc[fechas_incoherentes, 'release_year'].astype(int).astype(str)
                 + '-' + df_clean.loc[fechas_incoherentes, 'release_date'].dt.strftime('%m-%d'))
df_clean.loc[fechas_incoherentes, 'release_date'] = pd.to_datetime(fechas_nuevas)
fechas['fecha_corregida'] = df_clean.loc[fechas_incoherentes, 'release_date']
fechas.to_csv(RESULTADOS / 'fechas_corregidas.csv', index=False)

# Columnas críticas donde un valor de 0 o un nulo (NaN) es un error inaceptable
columnas_criticas = ['runtime', 'budget', 'revenue', 'vote_count', 'vote_average']

# Transformamos cualquier '0' en un nulo en esas columnas
df_clean[columnas_criticas] = df_clean[columnas_criticas].replace(0, np.nan)

# Eliminamos todas las filas que tengan un NaN o fecha nula
nulos_antes_drop = len(df_clean)
df_clean = df_clean.dropna(subset=columnas_criticas + ['release_date', 'popularity_level'])
# No convertimos una etiqueta ausente en una quinta clase llamada 'nan'.
filas_eliminadas_totales = nulos_antes_drop - len(df_clean)

print(f"Se eliminaron {filas_eliminadas_totales} filas por contener valores nulos o ceros")

# Conservar texto intacto y asegurar tipos numéricos enteros
df_clean['budget'] = df_clean['budget'].astype(int)
df_clean['revenue'] = df_clean['revenue'].astype(int)

# Resumen final
print(f"\nRESUMEN DE LIMPIEZA: {len(df_original)} registros iniciales -> {len(df_clean)} registros limpios finales.")

# Guardar el archivo final manteniendo el resto de columnas disponibles
df_clean.to_csv(CARPETA / 'IMBD_Movie_Dataset_limpio.csv', index=False)


# Tablas del EDA: dejamos constancia de los nulos y de los valores extremos.
df_original.isna().sum().rename('Nulos').to_csv(RESULTADOS / 'nulos_original.csv')
df_clean[['budget', 'revenue', 'runtime', 'vote_count', 'vote_average']].describe().to_csv(
    RESULTADOS / 'estadisticas.csv')
columnas_numericas = ['budget', 'revenue', 'runtime', 'vote_count', 'vote_average']
atipicos = []
for columna in columnas_numericas:
    q1 = df_clean[columna].quantile(0.25)
    q3 = df_clean[columna].quantile(0.75)
    iqr = q3 - q1
    mascara = (df_clean[columna] < q1 - 1.5 * iqr) | (df_clean[columna] > q3 + 1.5 * iqr)
    atipicos.append({'Variable': columna, 'Atípicos IQR': int(mascara.sum())})
pd.DataFrame(atipicos).to_csv(RESULTADOS / 'atipicos.csv', index=False)

# Los extremos de presupuesto o votos pueden ser películas válidas: los conservamos.
# StandardScaler prepara las escalas, pero no elimina ni comprime los extremos.
orden = ['Low', 'Medium', 'Moderately High', 'High']
fig, ax = plt.subplots(figsize=(8, 4))
df_clean['popularity_level'].value_counts().reindex(orden).plot.bar(ax=ax, color='#287c8e')
ax.set(title='Películas por nivel de popularidad', xlabel='Categoría', ylabel='Películas')
ax.tick_params(axis='x', rotation=0)
fig.tight_layout()
fig.savefig(GRAFICOS / 'categorias.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_clean['budget'] / 1_000_000, bins=30, color='#287c8e', edgecolor='white')
ax.set(title='Distribución del presupuesto', xlabel='Presupuesto (millones)', ylabel='Películas')
fig.tight_layout()
fig.savefig(GRAFICOS / 'presupuesto.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([df_clean.loc[df_clean['popularity_level'] == c, 'runtime'] for c in orden], tick_labels=orden)
ax.set(title='Duración por nivel de popularidad', xlabel='Categoría', ylabel='Minutos')
fig.tight_layout()
fig.savefig(GRAFICOS / 'duracion.png', dpi=150)
plt.close(fig)

generos = df_clean['genres'].fillna('').str.split('|').explode()
fig, ax = plt.subplots(figsize=(8, 4))
generos[generos != ''].value_counts().head(10).sort_values().plot.barh(ax=ax, color='#287c8e')
ax.set(title='Diez géneros más frecuentes', xlabel='Apariciones', ylabel='Género')
fig.tight_layout()
fig.savefig(GRAFICOS / 'generos.png', dpi=150)
plt.close(fig)

variables = ['budget', 'runtime', 'vote_count', 'vote_average', 'release_year', 'popularity']
correlacion = df_clean[variables].corr()
fig, ax = plt.subplots(figsize=(8, 6))
imagen = ax.imshow(correlacion, vmin=-1, vmax=1, cmap='RdBu_r')
ax.set_xticks(range(len(variables)), variables, rotation=45, ha='right')
ax.set_yticks(range(len(variables)), variables)
for i in range(len(variables)):
    for j in range(len(variables)):
        ax.text(j, i, f'{correlacion.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_title('Correlaciones: asociación, no causalidad')
fig.colorbar(imagen, ax=ax)
fig.tight_layout()
fig.savefig(GRAFICOS / 'correlaciones.png', dpi=150)
plt.close(fig)
print('Estadísticas y cinco gráficos del EDA guardados.')
