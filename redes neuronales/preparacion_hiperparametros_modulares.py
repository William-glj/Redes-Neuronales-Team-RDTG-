"""Compara ocho configuraciones del equipo de PyTorch sin usar prueba para elegir.

Ejecutar después de EDA.py: python preparacion_hiperparametros_modulares.py
"""

import copy
import json
import platform
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from EDA import CARPETA, CATEGORIAS, NOMBRES, marcar_entrenamiento_pendiente
from preparacion_datos_base import preparar_datos

EPOCAS = 15
SEMILLAS = [42, 7, 21]
EXPERIMENTOS = [
    {
        "nombre": "Exp_1",
        "capas": [
            32
        ],
        "batch": 32,
        "lr": 0.01
    },
    {
        "nombre": "Exp_2",
        "capas": [
            64,
            32
        ],
        "batch": 32,
        "lr": 0.01
    },
    {
        "nombre": "Exp_3",
        "capas": [
            128,
            64,
            32
        ],
        "batch": 32,
        "lr": 0.01
    },
    {
        "nombre": "Exp_4",
        "capas": [
            64,
            32
        ],
        "batch": 16,
        "lr": 0.01
    },
    {
        "nombre": "Exp_5",
        "capas": [
            64,
            32
        ],
        "batch": 64,
        "lr": 0.01
    },
    {
        "nombre": "Exp_6",
        "capas": [
            64,
            32
        ],
        "batch": 32,
        "lr": 0.001
    },
    {
        "nombre": "Exp_7",
        "capas": [
            64,
            32
        ],
        "batch": 32,
        "lr": 0.0001
    },
    {
        "nombre": "Exp_8",
        "capas": [
            64
        ],
        "batch": 32,
        "lr": 0.01
    }
]


class PeliculasNeuralNetwork(nn.Module):
    # Conservamos la arquitectura modular que ya tenía la rama desarrollo.
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def construir_red(entradas, ocultas):
    return PeliculasNeuralNetwork(entradas, ocultas, 4)


def entrenar(configuracion, semilla, x_entreno, y_entreno, x_val, y_val):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = construir_red(x_entreno.shape[1], configuracion["capas"])
    perdida = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=configuracion["lr"])
    datos = TensorDataset(x_entreno, y_entreno)
    lotes = DataLoader(datos, batch_size=configuracion["batch"], shuffle=True)
    historial = []
    mejor_perdida = float("inf")
    mejor_estado = None
    mejor_epoca = 0
    inicio = time.perf_counter()

    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        for x_lote, y_lote in lotes:
            optimizador.zero_grad()
            salida = modelo(x_lote)
            error = perdida(salida, y_lote)
            error.backward()
            optimizador.step()

        modelo.eval()
        with torch.no_grad():
            salida_entreno = modelo(x_entreno)
            salida_val = modelo(x_val)
            error_entreno = perdida(salida_entreno, y_entreno).item()
            error_val = perdida(salida_val, y_val).item()
            acierto_entreno = (salida_entreno.argmax(1) == y_entreno).float().mean().item()
            acierto_val = (salida_val.argmax(1) == y_val).float().mean().item()

        historial.append({"epoca": epoca, "perdida_entreno": error_entreno,
                          "perdida_validacion": error_val,
                          "accuracy_entreno": acierto_entreno,
                          "accuracy_validacion": acierto_val})
        # Conservamos el estado con MENOR pérdida de validación.
        if error_val < mejor_perdida:
            mejor_perdida = error_val
            mejor_estado = copy.deepcopy(modelo.state_dict())
            mejor_epoca = epoca

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        pred_val = modelo(x_val).argmax(1).numpy()
        pred_entreno = modelo(x_entreno).argmax(1).numpy()
    fila = {
        "experimento": configuracion["nombre"], "semilla": semilla,
        "capas": "-".join(str(n) for n in configuracion["capas"]),
        "batch": configuracion["batch"], "lr": configuracion["lr"],
        "mejor_epoca": mejor_epoca,
        "accuracy_entreno": accuracy_score(y_entreno.numpy(), pred_entreno),
        "accuracy_validacion": accuracy_score(y_val.numpy(), pred_val),
        "f1_validacion": f1_score(y_val.numpy(), pred_val, average="macro", zero_division=0),
        "perdida_validacion": mejor_perdida,
        "segundos": time.perf_counter() - inicio,
        "parametros": sum(p.numel() for p in modelo.parameters()),
    }
    return modelo, fila, pd.DataFrame(historial)


def ejecutar():
    marcar_entrenamiento_pendiente()
    resultados = CARPETA / "resultados"
    modelos = resultados / "modelos"
    modelos.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    entradas, objetivos, partes, variables = preparar_datos()
    x_entreno, x_val, x_prueba = [torch.tensor(x) for x in entradas]
    y_entreno, y_val, y_prueba = [torch.tensor(y) for y in objetivos]
    from preparacion_datos_entrenamiento import ejecutar_baseline
    ejecutar_baseline(entradas, objetivos)
    filas = []

    # La partición es siempre la misma. Cambian la inicialización y el barajado.
    for config in EXPERIMENTOS:
        for semilla in SEMILLAS:
            modelo, fila, historial = entrenar(config, semilla, x_entreno, y_entreno, x_val, y_val)
            filas.append(fila)
            historial.to_csv(resultados / f"historial_{config['nombre']}_{semilla}.csv", index=False)
            torch.save(modelo.state_dict(), modelos / f"{config['nombre']}_{semilla}.pt")
            print(f"{config['nombre']} · semilla {semilla} · "
                  f"F1 validación {fila['f1_validacion']:.3f} · época {fila['mejor_epoca']}", flush=True)

    detalle = pd.DataFrame(filas)
    detalle.to_csv(resultados / "experimentos_detalle.csv", index=False)
    comparacion = detalle.groupby(["experimento", "capas", "batch", "lr", "parametros"],
                                  as_index=False).agg(
        accuracy_entreno=("accuracy_entreno", "mean"),
        accuracy_validacion=("accuracy_validacion", "mean"),
        f1_validacion=("f1_validacion", "mean"),
        f1_desviacion=("f1_validacion", "std"),
        segundos=("segundos", "mean"),
    )
    # Criterio declarado: mayor F1 macro medio de VALIDACIÓN.
    comparacion = comparacion.sort_values(["f1_validacion", "f1_desviacion"],
                                         ascending=[False, True]).reset_index(drop=True)
    comparacion.to_csv(resultados / "comparacion.csv", index=False)
    ganador = comparacion.iloc[0]["experimento"]
    config_ganadora = next(c for c in EXPERIMENTOS if c["nombre"] == ganador)

    # Solo ahora se utiliza prueba, para la configuración ya elegida.
    pruebas = []
    for semilla in SEMILLAS:
        modelo = construir_red(len(variables), config_ganadora["capas"])
        modelo.load_state_dict(torch.load(modelos / f"{ganador}_{semilla}.pt", weights_only=True))
        modelo.eval()
        with torch.no_grad():
            predicciones = modelo(x_prueba).argmax(1).numpy()
        pruebas.append({"semilla": semilla,
                        "accuracy": accuracy_score(objetivos[2], predicciones),
                        "f1_macro": f1_score(objetivos[2], predicciones, average="macro", zero_division=0)})
        if semilla == 42:
            pred_principal = predicciones
            # Se fija 42 de antemano; no se elige la semilla por su nota en prueba.
            torch.save(modelo.state_dict(), resultados / "mejor_modelo.pt")

    tabla_prueba = pd.DataFrame(pruebas)
    tabla_prueba.to_csv(resultados / "prueba_semillas.csv", index=False)
    matriz = confusion_matrix(objetivos[2], pred_principal, labels=range(4))
    pd.DataFrame(matriz, index=NOMBRES, columns=NOMBRES).to_csv(resultados / "matriz_confusion.csv")
    informe = classification_report(objetivos[2], pred_principal, labels=range(4),
                                    target_names=NOMBRES, output_dict=True, zero_division=0)
    pd.DataFrame(informe).T.to_csv(resultados / "metricas_clases.csv")
    peliculas_prueba = partes[2][["id", "original_title", "popularity_level"]].copy()
    peliculas_prueba["prediccion"] = [CATEGORIAS[i] for i in pred_principal]
    peliculas_prueba["acierto"] = objetivos[2] == pred_principal
    peliculas_prueba.to_csv(resultados / "predicciones_prueba.csv", index=False)

    # Referencias sencillas: misma información de entrada y mismos conjuntos.
    mayoritaria = int(np.bincount(objetivos[0]).argmax())
    logistica = LogisticRegression(max_iter=2000, random_state=42)
    logistica.fit(entradas[0], objetivos[0])
    referencias = []
    for nombre in ["Clase mayoritaria", "Regresión logística"]:
        if nombre == "Clase mayoritaria":
            pv = np.full(len(objetivos[1]), mayoritaria)
            pp = np.full(len(objetivos[2]), mayoritaria)
        else:
            pv = logistica.predict(entradas[1])
            pp = logistica.predict(entradas[2])
        referencias.append({"modelo": nombre,
                            "accuracy_validacion": accuracy_score(objetivos[1], pv),
                            "f1_validacion": f1_score(objetivos[1], pv, average="macro", zero_division=0),
                            "accuracy_prueba": accuracy_score(objetivos[2], pp),
                            "f1_prueba": f1_score(objetivos[2], pp, average="macro", zero_division=0)})
    pd.DataFrame(referencias).to_csv(resultados / "referencias.csv", index=False)
    resumen = {
        "experimento_elegido": ganador, "configuracion": config_ganadora,
        "numero_experimentos": len(EXPERIMENTOS), "epocas_maximas": EPOCAS, "semillas": SEMILLAS, "semilla_principal": 42,
        "criterio": "Mayor F1 macro medio en validación; desempate por menor desviación",
        "entradas": variables, "categorias": CATEGORIAS,
        "filas": {n: len(p) for n, p in zip(["entrenamiento", "validacion", "prueba"], partes)},
        "accuracy_prueba_media": float(tabla_prueba["accuracy"].mean()),
        "f1_prueba_media": float(tabla_prueba["f1_macro"].mean()),
        "f1_prueba_desviacion": float(tabla_prueba["f1_macro"].std()),
        "accuracy_prueba_principal": pruebas[0]["accuracy"],
        "f1_prueba_principal": pruebas[0]["f1_macro"],
        "versiones": {"python": platform.python_version(), "torch": torch.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__},
    }
    (resultados / "resumen_modelo.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    from generar_informe import generar_resultados
    generar_resultados()
    print(f"\nElegido por validación: {ganador}")
    print(f"F1 macro en prueba (media de 3 semillas): {resumen['f1_prueba_media']:.3f}")


if __name__ == "__main__":
    ejecutar()
