"""Exploración y limpieza del CSV de películas.

Ejecutar: python EDA.py
Sin matplotlib, se pueden comprobar solo los datos: python EDA.py --sin-graficos
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
CATEGORIAS = ["Low", "Medium", "Moderately High", "High"]
NOMBRES = ["Baja", "Media", "Media-alta", "Alta"]
NUMERICAS = ["budget", "revenue", "runtime", "vote_count", "vote_average", "release_year"]


def marcar_entrenamiento_pendiente():
    # Evita mostrar métricas de una ejecución anterior al cambiar los datos.
    resultados = CARPETA / "resultados"
    resultados.mkdir(exist_ok=True)
    (resultados / "resumen_modelo.json").unlink(missing_ok=True)
    (resultados / "resultados_modelos.qmd").write_text(
        "::: {.callout-warning}\n## Entrenamiento pendiente\n"
        "Este informe es un borrador. Ejecutar `python EDA.py` y "
        "`python preparacion_hiperparametros_modulares.py` antes de generar la entrega con Quarto.\n:::\n",
        encoding="utf-8")


def tabla_markdown(df):
    # Escribimos tablas sin necesitar una biblioteca adicional.
    cabecera = "| " + " | ".join(str(c) for c in df.columns) + " |\n"
    separador = "| " + " | ".join("---" for c in df.columns) + " |\n"
    filas = ""
    for fila in df.itertuples(index=False, name=None):
        filas += "| " + " | ".join(str(v) for v in fila) + " |\n"
    return cabecera + separador + filas


def limpiar_datos():
    resultados = CARPETA / "resultados"
    resultados.mkdir(exist_ok=True)
    marcar_entrenamiento_pendiente()
    original = pd.read_csv(CARPETA / "datos/peliculas_original.csv")
    df = original.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])

    # Guardamos el diagnóstico ANTES de cambiar o eliminar valores.
    diagnostico = pd.DataFrame({
        "variable": df.columns,
        "tipo": df.dtypes.astype(str).values,
        "nulos": df.isna().sum().values,
    })
    diagnostico.to_csv(resultados / "nulos_original.csv", index=False)

    duplicados = int(df.duplicated(subset="id").sum())
    df = df.drop_duplicates(subset="id").copy()
    # Conservamos la delimitación comercial de la rama desarrollo.
    # Es una selección de muestra: no significa que todos los importes bajos sean errores.
    antes_filtro = len(df)
    df = df[(df["budget"] > 10000) & (df["revenue"] > 10000)].copy()
    excluidas_umbral = antes_filtro - len(df)
    sin_categoria = int((~df["popularity_level"].isin(CATEGORIAS)).sum())
    # No inventamos la etiqueta de una película que no la tiene.
    df = df[df["popularity_level"].isin(CATEGORIAS)].copy()

    invalidos = []
    for columna in NUMERICAS:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
        df[columna] = df[columna].replace([np.inf, -np.inf], np.nan)
        if columna == "vote_average":
            mascara = (df[columna] < 0) | (df[columna] > 10)
        elif columna == "release_year":
            # La muestra es histórica. Comprobamos coherencia, no actualidad.
            mascara = (df[columna] < 1888) | (df[columna] > 2015)
        elif columna in ["runtime", "budget", "revenue"]:
            mascara = df[columna] <= 0
        else:
            mascara = df[columna] < 0
        invalidos.append({"variable": columna, "invalidos": int(mascara.sum())})
        df.loc[mascara, columna] = np.nan

    # Las medianas, si hicieran falta, se calcularán SOLO con entrenamiento.
    # Aquí conservamos los nulos numéricos para no mezclar los conjuntos.
    df["genres"] = df["genres"].fillna("").str.strip()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    atipicos = []
    for columna in NUMERICAS + ["popularity"]:
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr
        cantidad = int(((df[columna] < inferior) | (df[columna] > superior)).sum())
        atipicos.append({"variable": columna, "limite_inferior": round(inferior, 3),
                         "limite_superior": round(superior, 3), "atipicos": cantidad})

    pd.DataFrame(atipicos).to_csv(resultados / "atipicos.csv", index=False)
    pd.DataFrame(invalidos).to_csv(resultados / "valores_invalidos.csv", index=False)
    df[NUMERICAS + ["popularity"]].describe().round(3).to_csv(resultados / "estadisticas.csv")
    df.to_csv(CARPETA / "datos/peliculas_limpias.csv", index=False)

    clases = df["popularity_level"].value_counts().reindex(CATEGORIAS, fill_value=0)
    resumen = pd.DataFrame({"categoria": NOMBRES, "peliculas": clases.values})
    resumen.to_csv(resultados / "clases.csv", index=False)

    # Auditar los rangos no demuestra cómo se generaron las etiquetas originales.
    rangos = df.groupby("popularity_level")["popularity"].agg(["min", "max", "count"])
    rangos = rangos.reindex(CATEGORIAS).reset_index()
    rangos.round(6).to_csv(resultados / "rangos_categorias.csv", index=False)

    texto = f"""Partimos de **{len(original)} películas** y **{original.shape[1]} columnas**, contando
el índice exportado. Hay **{duplicados} identificadores duplicados** y
**{sin_categoria} película sin una categoría válida**. Eliminamos esa fila para
no inventar su respuesta. Conservamos **{len(df)} películas**.

Conservamos la selección de la rama `desarrollo`: presupuesto y recaudación
superiores a 10.000. El filtro excluye **{excluidas_umbral} películas**. Es una
delimitación de la muestra comercial, no una prueba de que los importes bajos
sean erróneos: también excluye películas independientes que pueden ser válidas.
Esto limita la representatividad y no garantiza que los importes restantes
sean correctos. La eliminación de la categoría ausente es un paso adicional.

No se han imputado medianas con toda la muestra. La preparación posterior
calcula las medianas y la normalización únicamente con entrenamiento.

### Valores nulos en el archivo recibido

{tabla_markdown(diagnostico)}

### Distribución de las clases

{tabla_markdown(resumen)}

### Valores extremos

El criterio IQR señala valores por debajo de Q1 − 1,5 × IQR o por encima de
Q3 + 1,5 × IQR. Señalar un valor no demuestra que sea un error.

{tabla_markdown(pd.DataFrame(atipicos)[["variable", "atipicos"]])}

Conservamos estos extremos. Aplicamos `np.log1p` a presupuesto y
número de votos antes de normalizar las cinco entradas originales. Así reducimos las diferencias de escala
sin borrar automáticamente las películas con cifras elevadas. La duración,
la nota y el año solo se normalizan. La recaudación se estudia en el EDA y no entra
al modelo. `popularity` se estudia en el EDA, pero
**no se utiliza como entrada del modelo**.
"""
    (resultados / "resumen_eda.qmd").write_text(texto, encoding="utf-8")
    print(f"Original: {len(original)} | Duplicados: {duplicados} | Fuera del umbral: {excluidas_umbral} | Sin categoría: {sin_categoria}")
    print(f"Películas limpias: {len(df)}")
    print(resumen.to_string(index=False))
    return df


def crear_graficos(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta = CARPETA / "graficos"
    carpeta.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.figsize": (8, 4.5), "font.size": 11})

    clases = df["popularity_level"].value_counts().reindex(CATEGORIAS)
    fig, ax = plt.subplots()
    ax.bar(NOMBRES, clases.values, color="#2878a0")
    ax.set(title="Películas por nivel de popularidad", ylabel="Número de películas")
    fig.tight_layout()
    fig.savefig(carpeta / "clases.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["budget"].dropna() / 1_000_000, bins=30, color="#2878a0")
    axes[0].set(title="Presupuesto original", xlabel="Millones (importe del CSV)", ylabel="Películas")
    axes[1].hist(np.log1p(df["budget"].dropna()), bins=30, color="#d38435")
    axes[1].set(title="Presupuesto transformado", xlabel="log(1 + presupuesto)", ylabel="Películas")
    fig.tight_layout()
    fig.savefig(carpeta / "presupuesto.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    grupos = [df.loc[df["popularity_level"] == c, "runtime"].dropna() for c in CATEGORIAS]
    ax.boxplot(grupos, tick_labels=NOMBRES)
    ax.set(title="Duración según la categoría", ylabel="Duración (minutos)")
    fig.tight_layout()
    fig.savefig(carpeta / "duracion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    generos = df["genres"].str.get_dummies(sep="|").sum().sort_values(ascending=False).head(10)
    generos.sort_values().plot.barh(ax=ax, color="#2878a0")
    ax.set(title="Los diez géneros más frecuentes", xlabel="Películas", ylabel="")
    fig.tight_layout()
    fig.savefig(carpeta / "generos.png", dpi=150)
    plt.close(fig)

    columnas = ["budget", "revenue", "runtime", "vote_count", "vote_average", "popularity"]
    correlaciones = df[columnas].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 6))
    imagen = ax.imshow(correlaciones, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(columnas)), columnas, rotation=45, ha="right")
    ax.set_yticks(range(len(columnas)), columnas)
    for i in range(len(columnas)):
        for j in range(len(columnas)):
            ax.text(j, i, f"{correlaciones.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(correlaciones.iloc[i, j]) > 0.55 else "black")
    ax.set_title("Correlaciones de Pearson en la muestra")
    fig.colorbar(imagen, ax=ax)
    fig.tight_layout()
    fig.savefig(carpeta / "correlaciones.png", dpi=150)
    plt.close(fig)
    print("Guardados cinco gráficos en graficos/.")


if __name__ == "__main__":
    peliculas = limpiar_datos()
    if "--sin-graficos" not in sys.argv:
        crear_graficos(peliculas)
