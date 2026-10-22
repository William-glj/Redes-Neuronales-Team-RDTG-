import pandas as pd
import numpy as np

# Cargamos los datos del dataset original
df_original = pd.read_csv('TMBD Movie Dataset.csv')
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

# Columnas críticas donde un valor de 0 o un nulo (NaN) es un error inaceptable
columnas_criticas = ['runtime', 'budget', 'revenue', 'vote_count', 'vote_average']

# Transformamos cualquier '0' en un nulo en esas columnas
df_clean[columnas_criticas] = df_clean[columnas_criticas].replace(0, np.nan)

# Eliminamos todas las filas que tengan un NaN o fecha nula
nulos_antes_drop = len(df_clean)
df_clean = df_clean.dropna(subset=columnas_criticas + ['release_date'])
filas_eliminadas_totales = nulos_antes_drop - len(df_clean)

print(f"Se eliminaron {filas_eliminadas_totales} filas por contener valores nulos o ceros")

# Conservar texto intacto y asegurar tipos numéricos enteros
df_clean['budget'] = df_clean['budget'].astype(int)
df_clean['revenue'] = df_clean['revenue'].astype(int)

# Resumen final
print(f"\nRESUMEN DE LIMPIEZA: {len(df_original)} registros iniciales -> {len(df_clean)} registros limpios finales.")

# Guardar el archivo final manteniendo el resto de columnas disponibles
df_clean.to_csv('IMBD_Movie_Dataset_limpio.csv', index=False)

