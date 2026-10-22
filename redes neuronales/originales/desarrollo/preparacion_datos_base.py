import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Cargar el dataset limpio generado en el EDA
df = pd.read_csv('IMBD_Movie_Dataset_limpio.csv')

# 2. Seleccionar las variables explicativas (X) y la variable objetivo (Y)
feature_cols = ['budget', 'runtime', 'vote_count', 'vote_average', 'release_year']
X = df[feature_cols].values

# Convertir la variable categórica 'popularity_level' a etiquetas numéricas enteras
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['popularity_level'].astype(str))

# 3. Dividir los datos en conjuntos de entrenamiento (80%) y validación (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalizar los datos numéricos (media 0 y desviación típica 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

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



