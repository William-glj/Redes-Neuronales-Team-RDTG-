"""Preparación de las cinco variables del proyecto de desarrollo.

Mantenemos el planteamiento del equipo y añadimos una prueba final independiente.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from EDA import CARPETA, CATEGORIAS

FEATURE_COLS = ["budget", "runtime", "vote_count", "vote_average", "release_year"]


def preparar_datos():
    df = pd.read_csv(CARPETA / "datos/peliculas_limpias.csv")
    # No hacemos astype(str) sobre las etiquetas: convertiría un nulo en "nan".
    df["objetivo"] = df["popularity_level"].map({c: i for i, c in enumerate(CATEGORIAS)})
    if df["objetivo"].isna().any():
        raise ValueError("Quedan categorías sin limpiar. Ejecutar EDA.py primero.")

    entreno, reserva = train_test_split(df, test_size=0.30, random_state=42,
                                       stratify=df["objetivo"])
    validacion, prueba = train_test_split(reserva, test_size=0.50, random_state=42,
                                         stratify=reserva["objetivo"])
    partes = [entreno, validacion, prueba]
    medianas = entreno[FEATURE_COLS].median()
    if medianas.isna().any():
        raise ValueError("Hay una variable sin datos en entrenamiento.")

    valores_numericos = []
    for parte in partes:
        valores = parte[FEATURE_COLS].fillna(medianas).copy()
        for columna in ["budget", "vote_count"]:
            valores[columna] = np.log1p(valores[columna])
        valores_numericos.append(valores)

    scaler = StandardScaler()
    x_entreno = scaler.fit_transform(valores_numericos[0]).astype(np.float32)
    x_val = scaler.transform(valores_numericos[1]).astype(np.float32)
    x_prueba = scaler.transform(valores_numericos[2]).astype(np.float32)
    entradas = [x_entreno, x_val, x_prueba]
    objetivos = [parte["objetivo"].to_numpy(dtype=np.int64) for parte in partes]
    if not all(np.isfinite(x).all() for x in entradas):
        raise ValueError("Hay entradas no finitas.")

    resultados = CARPETA / "resultados"
    resultados.mkdir(exist_ok=True)
    particiones = []
    for nombre, parte in zip(["entrenamiento", "validacion", "prueba"], partes):
        filas = parte[["id", "original_title", "popularity_level"]].copy()
        filas["conjunto"] = nombre
        particiones.append(filas)
    pd.concat(particiones).to_csv(resultados / "particiones.csv", index=False)
    pd.DataFrame({"variable": FEATURE_COLS, "mediana": medianas.values,
                  "media_transformada": scaler.mean_,
                  "desviacion_transformada": scaler.scale_}).to_csv(
                      resultados / "normalizacion.csv", index=False)
    pd.Series(FEATURE_COLS, name="entrada").to_csv(resultados / "variables_entrada.csv", index=False)
    return entradas, objetivos, partes, FEATURE_COLS
