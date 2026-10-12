import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Importamos los datos ya preparados desde tu fichero 'preparacion_datos_base.py'
# (Asegúrate de que en 'preparacion_datos_base.py' tengas definidos train_loader y y)
from preparacion_datos_base import train_loader, y

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
# INICIALIZACIÓN Y ENTRENAMIENTO (CON BACKWARD)
# ==========================================
input_dim = 5  # Las 5 variables numéricas
hidden_dim = 64
output_dim = len(np.unique(y))  # Número de clases de popularidad

model = PeliculasNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(model.parameters(), lr=0.01)


# Bucle de entrenamiento donde actúa el BACKWARD automático
model.train()
for epoch in range(10):
    for batch_X, batch_y in train_loader:
        
        # 1. Forward (Predicción)
        outputs = model(batch_X)
        
        # 2. Cálculo del error (Loss)
        loss = criterio(outputs, batch_y)
        
        # 3. BACKWARD y actualización de pesos
        optimizador.zero_grad()
        loss.backward()  # <--- Aquí se ejecuta el cálculo hacia atrás
        optimizador.step()
        
    print(f"Época {epoch+1} completada. Loss: {loss.item():.4f}"
          
          
          


)