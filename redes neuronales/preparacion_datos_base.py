import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

CARPETA = Path(__file__).resolve().parent

# 1. Cargar el dataset limpio generado en el EDA
df = pd.read_csv(CARPETA / 'IMBD_Movie_Dataset_limpio.csv')

# 2. Seleccionar las variables explicativas (X) y la variable objetivo (Y)
feature_cols = ['budget', 'runtime', 'vote_count', 'vote_average', 'release_year']
X = df[feature_cols].values

# Convertir la variable categórica 'popularity_level' a etiquetas numéricas enteras
label_encoder = LabelEncoder()
if df['popularity_level'].isna().any():
    raise ValueError('Hay etiquetas ausentes. Ejecutar EDA.py primero.')
y = label_encoder.fit_transform(df['popularity_level'])

# 3. Reservar prueba final y separar entrenamiento y validación (70/15/15).
indices = np.arange(len(df))
idx_train, idx_reserva = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)
idx_val, idx_test = train_test_split(idx_reserva, test_size=0.50, random_state=42, stratify=y[idx_reserva])
X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

# 4. Normalizar los datos numéricos (media 0 y desviación típica 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 5. Convertir los arrays de NumPy a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 'long' para clases categóricas

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 6. Crear los DataLoaders para agrupar los datos en lotes (batches)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("¡Preparación de datos completada con éxito!")
print(f"Lotes de entrenamiento listos: {len(train_loader)} (Total de registros: {len(X_train_tensor)})")
print(f"Lotes de validación listos: {len(val_loader)} (Total de registros: {len(X_val_tensor)})")




# Guardamos la separación para comprobar que no se repiten películas.
partes = []
for nombre, indices_parte in [('entrenamiento', idx_train), ('validacion', idx_val), ('prueba', idx_test)]:
    parte = df.iloc[indices_parte][['id', 'original_title', 'popularity_level']].copy()
    parte['conjunto'] = nombre
    partes.append(parte)
pd.concat(partes).to_csv(CARPETA / 'resultados/particiones.csv', index=False)
pd.DataFrame({'Variable': feature_cols, 'Media entrenamiento': scaler.mean_,
              'Desviación entrenamiento': scaler.scale_}).to_csv(CARPETA / 'resultados/normalizacion.csv', index=False)
