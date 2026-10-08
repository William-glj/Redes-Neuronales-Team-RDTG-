import pandas as pd
import numpy as np




# Cargamos los datos del dataset
df_original = pd.read_csv('TMBD Movie Dataset.csv')

print(f"Dimensiones iniciales: {df_original.shape[0]} filas, {df_original.shape[1]} columnas")

# Creacion de una copia de los datos para no alterar el archivo original
df = df_original.copy()


# Revisamos que peliculas estan duplicadas y eliminamos aquellas que lo estan
filas_duplicadas = df.duplicated(subset=['id']).sum()
if filas_duplicadas > 0:
    df = df.drop_duplicates(subset=['id'], keep='first')
    print(f"Se eliminaron {filas_duplicadas} filas duplicadas por ID.")
else:
    print("No se encontraron duplicados.")


#Limpieza de valores no validos(0)
#Revisamos los valores de budget/revenue en caso de ser 0 los damos como no validos y los eliminamos
budget_revenue = (df['budget'] > 10000) & (df['revenue'] > 10000)
df_clean = df[budget_revenue].copy()

filas_eliminadas_budget_revenue = len(df) - len(df_clean)
print(f"Se eliminaron {filas_eliminadas_budget_revenue} filas con presupuesto o recaudación incoherentes (<= $10,000).")


# Tratamiento de valores nulos

# En caso de faltar duracion de la pelicula, se añade como valor la media de las demas peliculas
if df_clean['runtime'].isnull().sum() > 0:
    mediana_runtime = df_clean['runtime'].median()
    df_clean['runtime'] = df_clean['runtime'].fillna(mediana_runtime)
    print(f"Nulos en 'runtime' imputados con la mediana: {mediana_runtime} min.")



# Convertir fechas a objeto datetime
df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce')
# En caso de faltar la fecha de estreno se elimina dicha pelicula
if df_clean['release_date'].isnull().sum() > 0:
    df_clean = df_clean.dropna(subset=['release_date'])
    print("Filas eliminadas por fecha de estreno nula.")

# Reemplazar valores nulos en columnas de texto por cadenas vacías
columnas_texto = ['tagline', 'overview', 'homepage', 'genres', 'keywords']
for col in columnas_texto:
    df_clean[col] = df_clean[col].fillna('')


# Asegurar enteros para variables cuantitativas clave
df_clean['budget'] = df_clean['budget'].astype(int)
df_clean['revenue'] = df_clean['revenue'].astype(int)
# Limpiar barras o espacios sobrantes si existen, genres
df_clean['genres'] = df_clean['genres'].astype(str).str.strip()

# Resumen final de la limpieza
print("\nRESUMEN DE LIMPIEZA: ")
print(f"Registros iniciales: {len(df_original)}")
print(f"Registros finales limpios: {len(df_clean)}")



# Guardar el DataFrame limpio en un nuevo archivo CSV
#df_clean.to_csv('IMBD_Movie_Dataset_limpio.csv', index=False)



