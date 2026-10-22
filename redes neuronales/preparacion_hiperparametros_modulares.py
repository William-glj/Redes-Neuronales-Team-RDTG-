import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

# Importar los datos desde tu archivo base
from preparacion_datos_base import X_train, y_train, X_val, y_val, X_test, y_test, df, idx_test, label_encoder

CARPETA = Path(__file__).resolve().parent
RESULTADOS = CARPETA / "resultados"
SEMILLAS = [42, 7, 21]
torch.set_num_threads(1)
(RESULTADOS / "modelos").mkdir(exist_ok=True)


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

        for semilla in SEMILLAS:
            print(f"Semilla: {semilla}")
            train_loader, val_loader = crear_dataloaders(
                X_train, y_train, X_val, y_val, config['batch_size']
            )

            torch.manual_seed(semilla)
            np.random.seed(semilla)
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
                "Semilla": semilla,
                "Capas Ocultas": str(config['hidden_layers']),
                "Nº Capas": len(config['hidden_layers']),
                "Batch Size": config['batch_size'],
                "Learning Rate": config['learning_rate'],
                "Train Loss Final": hist_train_loss[-1],
                "Val Loss Final": hist_val_loss[-1],
                "Val Accuracy (%)": hist_val_acc[-1] * 100,
                "Tiempo (s)": tiempo_total
            })

            historicos[f"{nombre_exp}_{semilla}"] = {
                "train_loss": hist_train_loss,
                "val_loss": hist_val_loss,
                "val_acc": hist_val_acc,
                "config": config
            }

            torch.save(model.state_dict(), RESULTADOS / 'modelos' / f'{nombre_exp}_{semilla}.pt')
            pd.DataFrame({'Época': range(1, epochs + 1), 'Loss Train': hist_train_loss,
                          'Loss Val': hist_val_loss, 'Accuracy Val (%)': np.array(hist_val_acc) * 100}).to_csv(
                              RESULTADOS / f'historial_{nombre_exp}_{semilla}.csv', index=False)

    df_detalle = pd.DataFrame(resultados)
    df_detalle.to_csv(RESULTADOS / 'experimentos_detalle.csv', index=False)
    columnas_grupo = ['ID', 'Capas Ocultas', 'Nº Capas', 'Batch Size', 'Learning Rate']
    columnas_media = ['Train Loss Final', 'Val Loss Final', 'Val Accuracy (%)', 'Tiempo (s)']
    df_resultados = df_detalle.groupby(columnas_grupo, as_index=False)[columnas_media].mean()
    desviaciones = df_detalle.groupby('ID')['Val Accuracy (%)'].std().rename('Desviación accuracy')
    df_resultados = df_resultados.merge(desviaciones, on='ID')
    df_resultados = df_resultados.sort_values(['Val Accuracy (%)', 'Val Loss Final'], ascending=[False, True])
    df_resultados.to_csv(RESULTADOS / 'comparacion.csv', index=False)
    (RESULTADOS / 'historicos.json').write_text(json.dumps(historicos), encoding='utf-8')
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
    # La selección utiliza solo la accuracy media de validación.
    ganador = df_res.iloc[0]['ID']
    config = grid[int(ganador.split('_')[1]) - 1]
    clases = [str(c) for c in label_encoder.classes_]
    pruebas = []
    for semilla in SEMILLAS:
        modelo = PeliculasNeuralNetwork(5, config['hidden_layers'], len(clases))
        modelo.load_state_dict(torch.load(RESULTADOS / 'modelos' / f'{ganador}_{semilla}.pt', weights_only=True))
        modelo.eval()
        with torch.no_grad():
            predichas = modelo(torch.tensor(X_test, dtype=torch.float32)).argmax(1).numpy()
        pruebas.append({'Semilla': semilla, 'Accuracy': accuracy_score(y_test, predichas),
                        'F1 macro': f1_score(y_test, predichas, average='macro', zero_division=0)})
        if semilla == 42:
            pred_principal = predichas
            torch.save(modelo.state_dict(), RESULTADOS / 'mejor_modelo.pt')
    pd.DataFrame(pruebas).to_csv(RESULTADOS / 'prueba_semillas.csv', index=False)
    matriz = confusion_matrix(y_test, pred_principal, labels=range(len(clases)))
    pd.DataFrame(matriz, index=clases, columns=clases).to_csv(RESULTADOS / 'matriz_confusion.csv')
    reporte = classification_report(y_test, pred_principal, labels=range(len(clases)),
                                    target_names=clases, output_dict=True, zero_division=0)
    pd.DataFrame(reporte).T.to_csv(RESULTADOS / 'metricas_prueba.csv')
    peliculas = df.iloc[idx_test][['id', 'original_title', 'popularity_level']].copy()
    peliculas['Predicción'] = label_encoder.inverse_transform(pred_principal)
    peliculas['Acierto'] = y_test == pred_principal
    peliculas.to_csv(RESULTADOS / 'predicciones_prueba.csv', index=False)

    # Referencias simples con los mismos conjuntos y las mismas cinco variables.
    logistica = LogisticRegression(max_iter=2000, random_state=42)
    logistica.fit(X_train, y_train)
    clase_mayoritaria = np.bincount(y_train).argmax()
    referencias = pd.DataFrame([
        {'Modelo': 'Clase mayoritaria', 'Accuracy prueba (%)': 100 * accuracy_score(y_test, np.full(len(y_test), clase_mayoritaria))},
        {'Modelo': 'Regresión logística', 'Accuracy prueba (%)': 100 * accuracy_score(y_test, logistica.predict(X_test))},
        {'Modelo': 'Red neuronal', 'Accuracy prueba (%)': 100 * np.mean([p['Accuracy'] for p in pruebas])}
    ])
    referencias.to_csv(RESULTADOS / 'referencias.csv', index=False)
    resumen = {'base': '0857c4b', 'experimento_elegido': ganador, 'configuracion': config,
               'configuraciones': len(grid), 'semillas': SEMILLAS, 'epocas': 15,
               'entrenamiento': len(y_train), 'validacion': len(y_val), 'prueba': len(y_test),
               'clases': clases, 'accuracy_prueba_media': float(np.mean([p['Accuracy'] for p in pruebas])),
               'f1_prueba_medio': float(np.mean([p['F1 macro'] for p in pruebas])),
               'accuracy_validacion_media': float(df_res.iloc[0]['Val Accuracy (%)']) / 100}
    (RESULTADOS / 'resumen_modelo.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Configuración elegida por validación:', ganador)
    print(referencias.to_string(index=False))
