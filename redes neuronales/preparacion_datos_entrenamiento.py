"""Modelo inicial de desarrollo: 5 entradas, dos capas de 64 y 10 épocas."""

import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from EDA import CARPETA, NOMBRES
from preparacion_datos_base import preparar_datos


class PeliculasNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        return self.output(out)


def ejecutar_baseline(entradas=None, objetivos=None):
    if entradas is None:
        entradas, objetivos, _, _ = preparar_datos()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x_train = torch.tensor(entradas[0])
    y_train = torch.tensor(objetivos[0])
    x_val = torch.tensor(entradas[1])
    y_val = torch.tensor(objetivos[1])
    model = PeliculasNeuralNetwork(input_dim=5, hidden_dim=64, output_dim=4)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    historial = []
    for epoch in range(1, 11):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizador.zero_grad()
            loss = criterio(model(batch_X), batch_y)
            loss.backward()
            optimizador.step()
        model.eval()
        with torch.no_grad():
            salida_train = model(x_train)
            salida_val = model(x_val)
            historial.append({"epoca": epoch,
                              "perdida_entreno": criterio(salida_train, y_train).item(),
                              "perdida_validacion": criterio(salida_val, y_val).item(),
                              "accuracy_entreno": (salida_train.argmax(1) == y_train).float().mean().item(),
                              "accuracy_validacion": (salida_val.argmax(1) == y_val).float().mean().item()})
    resultados = CARPETA / "resultados"
    pd.DataFrame(historial).to_csv(resultados / "historial_baseline.csv", index=False)
    reporte = classification_report(objetivos[1], salida_val.argmax(1).numpy(), labels=range(4),
                                    target_names=NOMBRES, output_dict=True, zero_division=0)
    pd.DataFrame(reporte).T.to_csv(resultados / "metricas_baseline_validacion.csv")
    resumen = {"epocas": 10, "capas": [64, 64], "batch": 32, "lr": 0.01,
               "accuracy_validacion": historial[-1]["accuracy_validacion"],
               "f1_validacion": reporte["macro avg"]["f1-score"]}
    (resultados / "baseline.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(f"Modelo inicial · accuracy validación: {resumen['accuracy_validacion']:.1%}", flush=True)
    return resumen


if __name__ == "__main__":
    ejecutar_baseline()
