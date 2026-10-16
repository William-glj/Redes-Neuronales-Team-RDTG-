import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Importamos los datos ya preparados desde tu fichero 'preparacion_datos_base.py'
from preparacion_datos_base import train_loader, val_loader, y

# ==========================================
# DEFINICIÓN DE LA CLASE DE LA RED NEURONAL
# ==========================================
class PeliculasNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PeliculasNeuralNetwork, self).__init__()
        
        # Capas de neuronas
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Camino hacia adelante (Forward Pass)
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output(out)
        return out


# ==========================================
# INICIALIZACIÓN
# ==========================================
input_dim = 5  # Las 5 variables numéricas
hidden_dim = 64
output_dim = len(np.unique(y))  # Número de clases de popularidad

model = PeliculasNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(model.parameters(), lr=0.01)


# ==========================================
# FUNCIÓN DE EVALUACIÓN (Definida antes de usarla)
# ==========================================
def evaluar(model, loader, criterio):
    """Devuelve (loss media, accuracy, etiquetas reales, predicciones) sobre un DataLoader."""
    model.eval()                      # modo evaluación
    loss_total, aciertos, total = 0.0, 0, 0
    todas_reales, todas_pred = [], []
 
    with torch.no_grad():             # no se calculan gradientes
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterio(outputs, batch_y)
 
            predichas = outputs.argmax(dim=1)   # la clase con la puntuación más alta
 
            loss_total += loss.item() * len(batch_y)   
            aciertos += (predichas == batch_y).sum().item()
            total += len(batch_y)
 
            todas_reales.extend(batch_y.tolist())
            todas_pred.extend(predichas.tolist())
 
    return loss_total / total, aciertos / total, todas_reales, todas_pred


# ==========================================
# BUCLE DE ENTRENAMIENTO Y VALIDACIÓN
# ==========================================
for epoch in range(10):
    model.train()
    loss_acumulada = 0.0
    
    for batch_X, batch_y in train_loader:
        # 1. Forward (Predicción)
        outputs = model(batch_X)
        
        # 2. Cálculo del error (Loss)
        loss = criterio(outputs, batch_y)
        
        # 3. BACKWARD y actualización de pesos
        optimizador.zero_grad()
        loss.backward()  
        optimizador.step()
        
        loss_acumulada += loss.item()
 
    loss_train = loss_acumulada / len(train_loader)   # media de la época

    # VALIDACIÓN Y PRINT (¡Ahora sí están dentro del bucle de la época!)
    loss_val, acc_val, _, _ = evaluar(model, val_loader, criterio)
 
    print(f"Época {epoch+1:>2} | Loss train: {loss_train:.4f} | "
          f"Loss val: {loss_val:.4f} | Accuracy val: {acc_val*100:.2f}%")