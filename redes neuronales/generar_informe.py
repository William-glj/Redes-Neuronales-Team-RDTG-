"""Actualiza el fragmento del informe con métricas reales y sus figuras."""

import json
import numpy as np
import pandas as pd
from EDA import CARPETA, NOMBRES, tabla_markdown


def generar_resultados():
    carpeta = CARPETA / "resultados"
    destino = carpeta / "resultados_modelos.qmd"
    if not (carpeta / "resumen_modelo.json").exists():
        destino.write_text("""::: {.callout-warning}
## Entrenamiento pendiente
Todavía no hay métricas de las redes. Este documento es un borrador y no es la
entrega final. Ejecutar `python EDA.py` y `python preparacion_hiperparametros_modulares.py`, y volver
a generar el HTML con `quarto render`.
:::
""", encoding="utf-8")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    resumen = json.loads((carpeta / "resumen_modelo.json").read_text(encoding="utf-8"))
    comparacion = pd.read_csv(carpeta / "comparacion.csv")
    referencias = pd.read_csv(carpeta / "referencias.csv")
    prueba = pd.read_csv(carpeta / "prueba_semillas.csv")
    ganador = resumen["experimento_elegido"]
    tabla = comparacion[["experimento", "capas", "batch", "lr", "parametros",
                         "accuracy_validacion", "f1_validacion", "f1_desviacion"]].copy()
    tabla.columns = ["Prueba", "Neuronas ocultas", "Lote", "LR", "Parámetros",
                     "Accuracy val.", "F1 macro val.", "Desv. F1"]
    tabla = tabla.round(4)
    referencias = referencias.round(4)
    historial = pd.read_csv(carpeta / f"historial_{ganador}_42.csv")
    matriz = pd.read_csv(carpeta / "matriz_confusion.csv", index_col=0)
    graficos = CARPETA / "graficos"
    graficos.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    orden = comparacion.sort_values("experimento")
    ax.bar(orden["experimento"], orden["f1_validacion"],
           yerr=orden["f1_desviacion"], capsize=5, color="#2878a0")
    ax.set(ylim=(0, 1), title="Comparación en validación", ylabel="F1 macro medio ± desviación")
    fig.tight_layout()
    fig.savefig(graficos / "comparacion.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for columna, etiqueta in [("perdida_entreno", "Entrenamiento"),
                              ("perdida_validacion", "Validación")]:
        axes[0].plot(historial["epoca"], historial[columna], label=etiqueta)
    axes[0].set(title=f"Pérdida de {ganador} (semilla 42)", xlabel="Época", ylabel="Entropía cruzada")
    axes[0].legend()
    for columna, etiqueta in [("accuracy_entreno", "Entrenamiento"),
                              ("accuracy_validacion", "Validación")]:
        axes[1].plot(historial["epoca"], historial[columna], label=etiqueta)
    axes[1].set(title="Evolución de los aciertos", xlabel="Época", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(graficos / "aprendizaje.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, nombre in zip(axes, ["Exp_2", "Exp_3", "Exp_7"]):
        curva = pd.read_csv(carpeta / f"historial_{nombre}_42.csv")
        ax.plot(curva["epoca"], curva["perdida_entreno"], label="Entrenamiento")
        ax.plot(curva["epoca"], curva["perdida_validacion"], label="Validación")
        ax.set(title=f"{nombre} · semilla 42", xlabel="Época")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Entropía cruzada")
    fig.tight_layout()
    fig.savefig(graficos / "sobreajuste.png", dpi=150)
    plt.close(fig)

    baseline = json.loads((carpeta / "baseline.json").read_text())
    curva_base = pd.read_csv(carpeta / "historial_baseline.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curva_base["epoca"], curva_base["perdida_entreno"], label="Entrenamiento")
    ax.plot(curva_base["epoca"], curva_base["perdida_validacion"], label="Validación")
    ax.set(title="Modelo inicial de dos capas de 64", xlabel="Época", ylabel="Entropía cruzada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(graficos / "baseline.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    imagen = ax.imshow(matriz.values, cmap="Blues")
    ax.set_xticks(range(4), NOMBRES)
    ax.set_yticks(range(4), NOMBRES)
    ax.set(xlabel="Categoría predicha", ylabel="Categoría real",
           title=f"Matriz de confusión en prueba · {ganador}, semilla 42")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matriz.iloc[i, j]), ha="center", va="center",
                    color="white" if matriz.iloc[i, j] > matriz.values.max() / 2 else "black")
    fig.colorbar(imagen, ax=ax)
    fig.tight_layout()
    fig.savefig(graficos / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    principal = prueba[prueba["semilla"] == 42].iloc[0]
    logistica = referencias[referencias["modelo"] == "Regresión logística"].iloc[0]
    diferencia = resumen["f1_prueba_media"] - logistica["f1_prueba"]
    if diferencia > 0:
        conclusion = (f"La red obtiene un F1 macro medio {diferencia:.3f} superior al de la "
                      "regresión logística en esta partición. Es una diferencia observada, "
                      "no una prueba de superioridad general ni de significación estadística.")
    else:
        conclusion = ("La red no supera el F1 macro de la regresión logística en esta partición. "
                      "La referencia sencilla también es competitiva. Usar una red cumple "
                      "el objetivo de experimentar, pero no garantiza un mejor resultado.")
    # Ejemplo de error concreto extraído de la matriz, sin inventar causas.
    errores = matriz.to_numpy().copy()
    np.fill_diagonal(errores, -1)
    real, predicha = np.unravel_index(errores.argmax(), errores.shape)
    if errores[real, predicha] > 0:
        comentario_error = (f"El error más frecuente de la matriz es confundir **{NOMBRES[real]}** "
                            f"con **{NOMBRES[predicha]}**, en **{errores[real, predicha]} películas**. "
                            "Esto describe el error; la matriz no demuestra su causa.")
    else:
        comentario_error = "No hay errores de clasificación en esta matriz de prueba."
    metricas_clases = pd.read_csv(carpeta / "metricas_clases.csv", index_col=0).iloc[:4]
    metricas_clases = metricas_clases[["precision", "recall", "f1-score", "support"]].round(4)
    metricas_clases = metricas_clases.reset_index().rename(columns={"index": "Categoría"})
    config = resumen["configuracion"]
    filas = resumen["filas"]
    segunda = comparacion.iloc[1]
    diferencia_validacion = comparacion.iloc[0]["f1_validacion"] - segunda["f1_validacion"]
    puntuaciones = comparacion.set_index("experimento")["f1_validacion"]
    explicacion = f"""La ventaja de {ganador} respecto a {segunda['experimento']} es de solo
**{diferencia_validacion:.4f} de F1 macro**. Las desviaciones entre semillas son
**{comparacion.iloc[0]['f1_desviacion']:.4f}** y **{segunda['f1_desviacion']:.4f}**,
respectivamente. La elección cumple nuestro criterio; tres inicializaciones
sobre una muestra no demuestran una superioridad general.

En los cambios controlados, pasar de Exp_1 a Exp_8 cambia el F1 medio en
**{puntuaciones['Exp_8'] - puntuaciones['Exp_1']:+.3f}**; de Exp_8 a Exp_2, en
**{puntuaciones['Exp_2'] - puntuaciones['Exp_8']:+.3f}**; reducir el lote de Exp_2 a Exp_4,
en **{puntuaciones['Exp_4'] - puntuaciones['Exp_2']:+.3f}**; aumentar el lote de Exp_2
a Exp_5, en **{puntuaciones['Exp_5'] - puntuaciones['Exp_2']:+.3f}**; y reducir la tasa de
aprendizaje de Exp_2 a Exp_6, en **{puntuaciones['Exp_6'] - puntuaciones['Exp_2']:+.3f}**.
Son diferencias descriptivas de este ensayo, con variabilidad entre semillas.

Exp_7, con tasa 0,0001, obtiene un F1 medio de validación de
**{puntuaciones['Exp_7']:.3f}** dentro del límite de 15 épocas. Su peor resultado
es compatible con un aprendizaje más lento: no demuestra que esa tasa sea mala
si se permite más entrenamiento. La red más profunda, Exp_3, obtiene
**{puntuaciones['Exp_3']:.3f}**, por lo que aumentar capas y neuronas tampoco
garantiza mejorar en este ensayo.
"""
    detalle = pd.read_csv(carpeta / "experimentos_detalle.csv")
    epocas_elegidas = detalle.loc[detalle["experimento"] == ganador, "mejor_epoca"].tolist()
    if all(epoca == resumen["epocas_maximas"] for epoca in epocas_elegidas):
        explicacion += ("\nLas tres ejecuciones elegidas conservan la última época disponible, "
                        f"la {resumen['epocas_maximas']}. Esto indica que el límite de entrenamiento puede estar "
                        "condicionando el resultado. Una mejora futura sería estudiar más "
                        "épocas en validación; no hemos cambiado el plan después de ver prueba.\n")
    texto = f"""## Modelo inicial del equipo

Conservamos el planteamiento de `preparacion_datos_entrenamiento.py`: cinco
entradas, dos capas ocultas de 64 neuronas, lote 32, Adam con tasa 0,01 y
10 épocas. Con las correcciones y la nueva separación de datos, la accuracy
de validación es **{baseline['accuracy_validacion']:.1%}** y el F1 macro de
validación es **{baseline['f1_validacion']:.3f}**.

![Pérdida del modelo inicial calculada durante la ejecución, sin valores escritos a mano.](graficos/baseline.png)

Las cifras anteriores del informe de desarrollo correspondían a otro reparto
de datos y no se comparan directamente con estas. Este modelo inicial se mide
solo en validación; reservamos prueba para la configuración elegida después.

## Comparación de configuraciones

La partición contiene **{filas['entrenamiento']} películas de entrenamiento**,
**{filas['validacion']} de validación** y **{filas['prueba']} de prueba**.
Hay **{len(resumen['entradas'])} variables de entrada**. Entrenamos **{len(comparacion)}
configuraciones** con tres semillas cada una: **{len(comparacion) * 3} entrenamientos**,
además del modelo inicial. Los datos
asignados a cada conjunto son siempre los mismos.

{tabla_markdown(tabla)}

![F1 macro de validación. Las barras muestran la desviación entre tres inicializaciones.](graficos/comparacion.png)

Elegimos **{ganador}**, con capas ocultas **{config['capas']}**, lote
**{config['batch']}** y tasa de aprendizaje **{config['lr']}**, porque tiene el
mayor F1 macro medio en validación. Si hay empate, preferimos la menor
desviación. Dentro de cada entrenamiento recuperamos la época de menor
pérdida de validación; no usamos la última época automáticamente.

Exp_1 frente a Exp_8 cambia las neuronas de una capa. Exp_8 frente a Exp_2
añade una segunda capa manteniendo 64 neuronas en la primera. Exp_2 frente a
Exp_3 cambia profundidad y anchura a la vez, por lo que no aislamos sus efectos.
Exp_4 y Exp_5 cambian solo el lote respecto a Exp_2. Exp_6 y Exp_7 cambian
solo la tasa de aprendizaje respecto a Exp_2.

{explicacion}

## Curvas de aprendizaje

![Pérdida y accuracy durante las 15 épocas de la configuración elegida.](graficos/aprendizaje.png)

La curva corresponde a la semilla 42, fijada de antemano. Si la pérdida de
entrenamiento baja mientras la de validación sube, hay indicios de
sobreajuste. Las curvas muestran todas las épocas, aunque recuperamos el
estado de menor pérdida de validación.

![Comparación de las pérdidas de tres configuraciones, manteniendo la semilla 42.](graficos/sobreajuste.png)

Estas curvas permiten ver si una mayor complejidad o un aprendizaje más rápido
reducen la pérdida de entrenamiento sin mejorar validación. Se muestran todas
las épocas, pero las métricas comparadas corresponden a los pesos recuperados
por validación. Las curvas no justifican elegir una red por su acierto de
entrenamiento.

## Evaluación final en prueba

Consultamos prueba después de elegir la configuración. Para las tres semillas,
la accuracy media es **{resumen['accuracy_prueba_media']:.1%}** y el F1 macro
medio es **{resumen['f1_prueba_media']:.3f} ± {resumen['f1_prueba_desviacion']:.3f}**.
La desviación resume tres inicializaciones sobre UNA partición; no es un
intervalo de confianza ni mide la variación entre distintas muestras.

El F1 medio de validación de la configuración elegida es
**{comparacion.iloc[0]['f1_validacion']:.3f}**. Su descenso hasta
**{resumen['f1_prueba_media']:.3f}** en prueba muestra que la estimación de
validación fue más favorable. La muestra reservada es pequeña y la selección
entre configuraciones también puede favorecer resultados optimistas en
validación. No retocamos el modelo tras observar esta prueba.

{tabla_markdown(prueba.round(4))}

Para la matriz y los ejemplos usamos siempre la semilla 42: accuracy
**{principal['accuracy']:.1%}** y F1 macro **{principal['f1_macro']:.3f}**.

![La diagonal contiene los aciertos; fuera de ella están los errores.](graficos/matriz_confusion.png)

{comentario_error}

{tabla_markdown(metricas_clases)}

## Comparación con referencias sencillas

La clase mayoritaria se decide usando entrenamiento. La regresión logística
usa las mismas entradas normalizadas y es un clasificador, a pesar de su
nombre. No hemos ajustado sus hiperparámetros. Estas referencias ayudan a
valorar si la red aporta algo frente a soluciones menos complejas.

{tabla_markdown(referencias)}

## Conclusiones de los resultados

1. **{ganador}** es la configuración seleccionada por validación en este ensayo.
   Eso no implica que sea la mejor red para cualquier dataset.
2. {conclusion}
3. La accuracy mide la proporción de aciertos. El F1 macro combina precisión
   y exhaustividad y da el mismo peso a cada categoría. Por eso lo utilizamos
   como criterio de comparación.
4. Tres semillas ayudan a comprobar la sensibilidad a la inicialización.
   Para conclusiones más sólidas harían falta otras particiones y más datos.

Los CSV detallados, las predicciones, los parámetros de normalización y los
pesos se guardan en `resultados/`. Las versiones utilizadas son:
`{resumen['versiones']}`.
"""
    destino.write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    generar_resultados()
