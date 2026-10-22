import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from preparacion_datos_base import train_loader, val_loader, y, label_encoder

# ==========================================
# DEFINICIÓN DE LA CLASE DE LA RED NEURONAL
# ==========================================
class PeliculasNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PeliculasNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output(out)
        return out


# ==========================================
# INICIALIZACIÓN
# ==========================================
input_dim = 5
hidden_dim = 64
output_dim = len(np.unique(y))

model = PeliculasNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(model.parameters(), lr=0.01)


# ==========================================
# FUNCIÓN DE EVALUACIÓN 
# ==========================================
def evaluar(model, loader, criterio):
    model.eval()
    loss_total, aciertos, total = 0.0, 0, 0
    todas_reales, todas_pred = [], []
 
    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterio(outputs, batch_y)
 
            predichas = outputs.argmax(dim=1)
 
            loss_total += loss.item() * len(batch_y)   
            aciertos += (predichas == batch_y).sum().item()
            total += len(batch_y)
 
            todas_reales.extend(batch_y.tolist())
            todas_pred.extend(predichas.tolist())
 
    return loss_total / total, aciertos / total, todas_reales, todas_pred


# ==========================================
# BUCLE DE ENTRENAMIENTO Y VALIDACIÓN
# ==========================================
print("=== ENTRENANDO MODELO INICIAL (BASELINE) ===")
for epoch in range(10):
    model.train()
    loss_acumulada = 0.0
    
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterio(outputs, batch_y)
        
        optimizador.zero_grad()
        loss.backward()  
        optimizador.step()
        
        loss_acumulada += loss.item() * len(batch_y)
 
    loss_train = loss_acumulada / len(train_loader.dataset)
    loss_val, acc_val, _, _ = evaluar(model, val_loader, criterio)
 
    print(f"Época {epoch+1:>2} | Loss train: {loss_train:.4f} | "
          f"Loss val: {loss_val:.4f} | Accuracy val: {acc_val*100:.2f}%")

# ==========================================
# BLOQUE DE PRUEBA Y MÉTRICAS FINALES
# ==========================================
print("\n" + "="*50)
print("RESULTADOS DE EVALUACIÓN DEL MODELO BASELINE")
print("="*50)

loss_val, acc_val, y_reales, y_predichas = evaluar(model, val_loader, criterio)

# Obtener los nombres de las clases originales
nombres_clases = [str(cls) for cls in label_encoder.classes_]
todas_las_etiquetas = list(range(len(nombres_clases)))

print(f"\nAccuracy Final en Validación: {acc_val * 100:.2f}%\n")
print("Reporte de Clasificación por Clase:")
print(classification_report(
    y_reales, 
    y_predichas, 
    labels=todas_las_etiquetas,
    target_names=nombres_clases,
    zero_division=0
))