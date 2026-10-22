import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Importar los datos desde tu archivo base
from preparacion_datos_base import X_train, y_train, X_val, y_val


# ==============================================================================
# 1. ARQUITECTURA FLEXIBLE DE RED NEURONAL
# ==============================================================================
class PeliculasNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation=nn.ReLU):
        """
        hidden_layers: lista de enteros, ej: [128, 64] -> 2 capas (128 y 64 neuronas)
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        
        # Construcción dinámica de capas ocultas
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
            
        # Capa final de salida
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# 2. FUNCIONES AUXILIARES: CREACIÓN DE LOADERS Y EVALUACIÓN
# ==============================================================================
def crear_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    """Recrea los DataLoaders según el batch_size indicado."""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def evaluar(model, loader, criterio):
    """Evalúa el modelo sobre un DataLoader y devuelve loss y accuracy."""
    model.eval()
    loss_total, aciertos, total = 0.0, 0, 0
 
    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterio(outputs, batch_y)
            predichas = outputs.argmax(dim=1)
 
            loss_total += loss.item() * len(batch_y)   
            aciertos += (predichas == batch_y).sum().item()
            total += len(batch_y)
 
    return loss_total / total, aciertos / total


# ==============================================================================
# 3. BUCLE DE EXPERIMENTACIÓN SISTEMÁTICA
# ==============================================================================
def ejecutar_experimentos(X_train, y_train, X_val, y_val, grid_experimentos, epochs=15):
    # Asegurar que los datos sean Tensors de PyTorch con el tipo correcto
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(np.array(y_val), dtype=torch.long)

    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))
    
    resultados = []
    historicos = {}

    for i, config in enumerate(grid_experimentos):
        nombre_exp = f"Exp_{i+1}"
        print(f"\n[{nombre_exp}/{len(grid_experimentos)}] Probando: {config}")
        
        train_loader, val_loader = crear_dataloaders(
            X_train, y_train, X_val, y_val, config['batch_size']
        )
        
        model = PeliculasNeuralNetwork(
            input_dim=input_dim,
            hidden_layers=config['hidden_layers'],
            output_dim=output_dim
        )
        criterio = nn.CrossEntropyLoss()
        optimizador = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        hist_train_loss, hist_val_loss, hist_val_acc = [], [], []
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            loss_acum = 0.0
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterio(outputs, batch_y)
                
                optimizador.zero_grad()
                loss.backward()
                optimizador.step()
                
                loss_acum += loss.item() * len(batch_y)
            
            loss_train = loss_acum / len(train_loader.dataset)
            loss_val, acc_val = evaluar(model, val_loader, criterio)
            
            hist_train_loss.append(loss_train)
            hist_val_loss.append(loss_val)
            hist_val_acc.append(acc_val)
            
        tiempo_total = time.time() - start_time
        
        resultados.append({
            "ID": nombre_exp,
            "Capas Ocultas": str(config['hidden_layers']),
            "Nº Capas": len(config['hidden_layers']),
            "Batch Size": config['batch_size'],
            "Learning Rate": config['learning_rate'],
            "Train Loss Final": round(hist_train_loss[-1], 4),
            "Val Loss Final": round(hist_val_loss[-1], 4),
            "Val Accuracy (%)": round(hist_val_acc[-1] * 100, 2),
            "Tiempo (s)": round(tiempo_total, 2)
        })
        
        historicos[nombre_exp] = {
            "train_loss": hist_train_loss,
            "val_loss": hist_val_loss,
            "val_acc": hist_val_acc,
            "config": config
        }

    df_resultados = pd.DataFrame(resultados).sort_values(by="Val Accuracy (%)", ascending=False)
    return df_resultados, historicos


# ==============================================================================
# 4. EJECUCIÓN DEL GRID
# ==============================================================================
if __name__ == "__main__":
    grid = [

        {"hidden_layers": [32], "batch_size": 32, "learning_rate": 0.01},
        {"hidden_layers": [64, 32], "batch_size": 32, "learning_rate": 0.01},
        {"hidden_layers": [128, 64, 32], "batch_size": 32, "learning_rate": 0.01},
        {"hidden_layers": [64, 32], "batch_size": 16, "learning_rate": 0.01},
        {"hidden_layers": [64, 32], "batch_size": 64, "learning_rate": 0.01},
        {"hidden_layers": [64, 32], "batch_size": 32, "learning_rate": 0.001},
        {"hidden_layers": [64, 32], "batch_size": 32, "learning_rate": 0.0001},

        ]
    #2 modelo de pruebas.Excede umbrales
    #grid = [
    # 1. Variación de Arquitectura (Base: batch=32, lr=0.01)
    #{"hidden_layers": [32], "batch_size": 32, "learning_rate": 0.01},
    #{"hidden_layers": [64, 32], "batch_size": 32, "learning_rate": 0.01},
    #{"hidden_layers": [128, 64, 32], "batch_size": 32, "learning_rate": 0.01},
    #{"hidden_layers": [256, 128, 64, 32], "batch_size": 32, "learning_rate": 0.01},  # Red compleja (4 capas)

    # 2. Variación de Tamaño de Lote (Base: [64, 32], lr=0.01)
    #{"hidden_layers": [64, 32], "batch_size": 16, "learning_rate": 0.01},
    #{"hidden_layers": [64, 32], "batch_size": 64, "learning_rate": 0.01},

    # 3. Variación de Tasa de Aprendizaje (Base: [64, 32], batch=32)
    #{"hidden_layers": [64, 32], "batch_size": 32, "learning_rate": 0.001},
    #{"hidden_layers": [64, 32], "batch_size": 32, "learning_rate": 0.0001},

    # 4. Combinaciones Cruzadas Interesantes
    #{"hidden_layers": [128, 64, 32], "batch_size": 32, "learning_rate": 0.0001},
    #{"hidden_layers": [256, 128, 64, 32], "batch_size": 32, "learning_rate": 0.001},
    #]



    df_res, historico_exp = ejecutar_experimentos(
        X_train, y_train, X_val, y_val, grid_experimentos=grid, epochs=15
    )

    print("\n=== RESULTADOS DEL GRID SEARCH ===")
    print(df_res.to_string(index=False))