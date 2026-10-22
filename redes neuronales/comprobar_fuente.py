"""Compara el CSV recibido con el ZIP público de Kaggle, sin cambiar los datos.

Uso: python comprobar_fuente.py /ruta/al/archivo.zip
Descarga: https://www.kaggle.com/datasets/juzershakir/tmdb-movies-dataset
"""

from pathlib import Path
import hashlib
import json
import sys
import zipfile
import numpy as np
import pandas as pd
CARPETA = Path(__file__).resolve().parent
CATEGORIAS = ["Low", "Medium", "Moderately High", "High"]


def comprobar_fuente(ruta_zip):
    with zipfile.ZipFile(ruta_zip) as archivo:
        contenido = archivo.read("tmdb_movies_data.csv")
        with archivo.open("tmdb_movies_data.csv") as csv:
            publico = pd.read_csv(csv)

    recibido = pd.read_csv(CARPETA / "TMBD Movie Dataset.csv")
    seleccion = publico.copy()
    seleccion[["budget", "revenue"]] = seleccion[["budget", "revenue"]].replace(0, np.nan)
    seleccion = seleccion.dropna().drop_duplicates(subset="id")
    if set(seleccion["id"]) != set(recibido["id"]):
        raise ValueError("La selección del dataset público no coincide con los identificadores recibidos.")

    comparacion = recibido.merge(seleccion, on="id", suffixes=("_recibido", "_publico"),
                                 validate="one_to_one")
    verificadas = []
    for columna in publico.columns:
        if columna == "id":
            verificadas.append(columna)
            continue
        a = comparacion[columna + "_recibido"]
        b = comparacion[columna + "_publico"]
        if columna == "release_date":
            fechas_recibidas = pd.to_datetime(a, format="%Y-%m-%d")
            fechas_publicas = pd.to_datetime(b, format="%m/%d/%Y")
            fechas_incoherentes = int((fechas_recibidas.dt.year != comparacion["release_year_recibido"]).sum())
            # La copia heredada cambia el siglo en 12 fechas. El año separado sí es correcto.
            fechas_corregidas = comparacion["release_year_recibido"].astype(int).astype(str) + "-" + fechas_recibidas.dt.strftime("%m-%d")
            iguales = pd.to_datetime(fechas_corregidas, format="%Y-%m-%d").equals(fechas_publicas)
        elif columna in ["budget_adj", "revenue_adj"]:
            # El descargable tiene menos decimales en estas dos columnas.
            # No entran en nuestra red. Permitimos un error relativo de 1e-9.
            iguales = np.allclose(a, b, rtol=1e-9, atol=1e-6)
        elif pd.api.types.is_numeric_dtype(a):
            # Dos CSV pueden representar el mismo decimal con distinto redondeo binario.
            iguales = np.allclose(a, b, rtol=1e-12, atol=1e-12)
        else:
            iguales = a.equals(b)
        if not iguales:
            raise ValueError("Hay diferencias no explicadas en " + columna)
        verificadas.append(columna)

    if not np.array_equal(recibido["profit"], recibido["revenue"] - recibido["budget"]):
        raise ValueError("La columna profit no coincide con revenue - budget.")

    cortes = recibido["popularity"].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    etiquetas = pd.cut(recibido["popularity"], bins=cortes, labels=CATEGORIAS)
    iguales = etiquetas.astype("string").fillna("SIN_ETIQUETA").equals(
        recibido["popularity_level"].astype("string").fillna("SIN_ETIQUETA"))
    if not iguales:
        raise ValueError("Las etiquetas recibidas no se reproducen con estos cuartiles.")

    resumen = {
        "fuente_publica": "https://www.kaggle.com/datasets/juzershakir/tmdb-movies-dataset",
        "descarga": "https://www.kaggle.com/api/v1/datasets/download/juzershakir/tmdb-movies-dataset",
        "publicador": "Juzer Shakir",
        "version_kaggle": 1,
        "licencia_indicada_en_la_ficha": "Unknown",
        "sha256_csv_publico": hashlib.sha256(contenido).hexdigest(),
        "sha256_csv_recibido": hashlib.sha256(
            (CARPETA / "TMBD Movie Dataset.csv").read_bytes()).hexdigest(),
        "filas_publicas": len(publico),
        "filas_seleccion_verificada": len(seleccion),
        "columnas_verificadas": verificadas,
        "tolerancia_numerica_general": "rtol=1e-12, atol=1e-12",
        "columnas_con_tolerancia_relativa_1e_9": ["budget_adj", "revenue_adj"],
        "fechas_recibidas_con_siglo_incorrecto": fechas_incoherentes,
        "cortes_popularidad": cortes,
        "etiquetas_reproducidas_incluyendo_el_nulo": len(recibido),
        "nota": "Verifica una fuente pública equivalente; no acredita desde qué enlace descargó el equipo su copia.",
    }
    destino = CARPETA / "datos/fuente_verificada.json"
    destino.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verificados {len(recibido)} identificadores y {len(verificadas)} columnas.")
    print("Las categorías y profit también se reproducen. No se ha modificado el CSV recibido.")
    print("Comprobación guardada en datos/fuente_verificada.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python comprobar_fuente.py /ruta/al/archivo.zip")
    comprobar_fuente(Path(sys.argv[1]))
