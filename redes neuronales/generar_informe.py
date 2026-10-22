"""Genera tablas y figuras del informe a partir de las ejecuciones guardadas."""
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rdtg-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CARPETA = Path(__file__).resolve().parent
RESULTADOS = CARPETA / 'resultados'
GRAFICOS = CARPETA / 'graficos'


def tabla(datos):
    columnas = [str(c) for c in datos.columns]
    lineas = ['| ' + ' | '.join(columnas) + ' |', '| ' + ' | '.join(['---'] * len(columnas)) + ' |']
    for fila in datos.itertuples(index=False, name=None):
        valores = [str(v).replace('|', ' / ').replace('\n', ' ') for v in fila]
        lineas.append('| ' + ' | '.join(valores) + ' |')
    return '\n'.join(lineas)


def guardar(nombre, texto):
    (RESULTADOS / (nombre + '.qmd')).write_text(texto.rstrip() + '\n', encoding='utf-8')


def generar():
    datos = pd.read_csv(CARPETA / 'IMBD_Movie_Dataset_limpio.csv')
    original = pd.read_csv(CARPETA / 'TMBD Movie Dataset.csv')
    resumen = json.loads((RESULTADOS / 'resumen_modelo.json').read_text())
    baseline = json.loads((RESULTADOS / 'baseline.json').read_text())
    comparacion = pd.read_csv(RESULTADOS / 'comparacion.csv')
    historial = pd.read_csv(RESULTADOS / 'historial_baseline.csv')
    metricas = pd.read_csv(RESULTADOS / 'metricas_baseline.csv', index_col=0)
    clases = resumen['clases']

    guardar('muestra', tabla(datos[['id', 'original_title', 'release_date', 'budget', 'revenue', 'runtime']].head(4)))
    estadisticas = datos[['budget', 'revenue', 'runtime']].describe().round(2).reset_index()
    guardar('estadisticas', tabla(estadisticas))
    nulos = original.isna().sum().rename('Nulos recibidos').reset_index()
    nulos.columns = ['Columna', 'Nulos recibidos']
    atipicos = pd.read_csv(RESULTADOS / 'atipicos.csv')
    guardar('diagnostico', '#### Valores nulos\n\n' + tabla(nulos) + '\n\n#### Valores atípicos\n\n' + tabla(atipicos))
    conteos = datos['popularity_level'].value_counts().reindex(['Low', 'Medium', 'Moderately High', 'High'])
    guardar('limpieza', f'Partimos de **{len(original)} películas** y quedan **{len(datos)}**: '
            'se excluyen diez por el filtro comercial y una sin categoría. '
            'No se encontraron duplicados por ID. Se corrigen doce fechas con el siglo incorrecto, '
            'conservando el año de estreno ya contrastado.\n\n' + tabla(conteos.rename('Películas').reset_index()))

    guardar('resumen_baseline', f'::: {{.callout-note}}\n**Resumen de Datos**\n\n'
            f'- Registros de entrenamiento: **{baseline["entrenamiento"]}**.\n'
            f'- Registros de validación: **{baseline["validacion"]}**.\n'
            f'- Registros de prueba reservados: **{resumen["prueba"]}**.\n'
            f'- Épocas del modelo inicial: **{baseline["epocas"]}**.\n'
            f'- Accuracy final en validación: **{baseline["accuracy_validacion"] * 100:.2f} %**.\n:::\n\n'
            'Las cifras siguientes se generan desde los CSV de esta ejecución. No corresponden a la antigua partición 80/20.')
    guardar('historial_baseline', tabla(historial.round(4)))
    guardar('metricas_baseline', tabla(metricas.drop(index='accuracy').round(3).reset_index().rename(columns={'index': 'Clase'})))
    guardar('comparacion', tabla(comparacion.round(4)) + '\n\nLas pérdidas, accuracy y tiempos son medias de las semillas 42, 7 y 21. '
            'La desviación se expresa en puntos porcentuales de accuracy. El tiempo depende del equipo.')

    primero = comparacion.iloc[0]
    segundo = comparacion.iloc[1]
    configuraciones = comparacion.set_index('ID')
    conclusiones = f'''La mejor configuración según el criterio fijado es **{primero['ID']}**: capas ocultas
**{primero['Capas Ocultas']}**, lote **{int(primero['Batch Size'])}** y tasa de aprendizaje
**{primero['Learning Rate']}**. Su accuracy media de validación es **{primero['Val Accuracy (%)']:.2f} %**.
Se selecciona con validación; no se utiliza prueba para escogerla.

1. **Tamaño del lote:** Exp_2, Exp_4 y Exp_5 mantienen la red [64, 32] y la tasa 0,01.
   Sus accuracies medias son {configuraciones.loc['Exp_2', 'Val Accuracy (%)']:.2f} %, {configuraciones.loc['Exp_4', 'Val Accuracy (%)']:.2f} % y {configuraciones.loc['Exp_5', 'Val Accuracy (%)']:.2f} %.
   La comparación describe estos experimentos; no demuestra una causa universal del resultado.
2. **Tasa de aprendizaje:** Exp_2, Exp_6 y Exp_7 usan la misma arquitectura y lote.
   Con tasa 0,0001, Exp_7 obtiene {configuraciones.loc['Exp_7', 'Val Accuracy (%)']:.2f} %.
   Con quince épocas puede necesitar más entrenamiento; no significa que esa tasa nunca funcione.
3. **Profundidad y neuronas:** las redes [32], [64, 32] y [128, 64, 32] cambian capas y anchura.
   No aislamos el efecto de cada cambio. La red mayor consigue {configuraciones.loc['Exp_3', 'Val Accuracy (%)']:.2f} %.
   Más neuronas no garantizan más aciertos. Para estudiar sobreajuste miramos las curvas completas.

La diferencia entre {primero['ID']} y {segundo['ID']} es de **{primero['Val Accuracy (%)'] - segundo['Val Accuracy (%)']:.2f} puntos**.
La desviación de {primero['ID']} entre semillas es **{primero['Desviación accuracy']:.2f} puntos**.
Tres repeticiones ayudan a ver variación, pero no demuestran significación estadística.
'''
    guardar('conclusiones', conclusiones)
    pruebas = pd.read_csv(RESULTADOS / 'prueba_semillas.csv')
    referencias = pd.read_csv(RESULTADOS / 'referencias.csv')
    metricas_prueba = pd.read_csv(RESULTADOS / 'metricas_prueba.csv', index_col=0)
    predicciones = pd.read_csv(RESULTADOS / 'predicciones_prueba.csv')
    ejemplos = pd.concat([predicciones[predicciones['Acierto']].head(2), predicciones[~predicciones['Acierto']].head(2)])
    ejemplos = ejemplos[['original_title', 'popularity_level', 'Predicción', 'Acierto']]
    ejemplos.to_csv(RESULTADOS / 'ejemplos.csv', index=False)
    logistica = referencias.loc[referencias['Modelo'] == 'Regresión logística', 'Accuracy prueba (%)'].iloc[0]
    diferencia = resumen['accuracy_prueba_media'] * 100 - logistica
    principal = pruebas.loc[pruebas['Semilla'] == 42].iloc[0]
    guardar('prueba', f'''La configuración ya elegida obtiene **{resumen['accuracy_prueba_media'] * 100:.2f} % de accuracy medio en prueba**
y **F1 macro {resumen['f1_prueba_medio']:.3f}**, promediando las tres semillas.
Accuracy es aciertos divididos entre películas. F1 macro combina precisión y recall y promedia las cuatro clases con igual peso.

{tabla(pruebas.round(4))}

La matriz usa la semilla 42, fijada de antemano: **{round(principal['Accuracy'] * resumen['prueba'])} aciertos de {resumen['prueba']} películas**.
Las filas son las categorías reales y las columnas las predichas; la diagonal representa aciertos.

![Matriz de confusión de prueba, semilla 42.](graficos/matriz_confusion.png)

{tabla(metricas_prueba.drop(index='accuracy').round(3).reset_index().rename(columns={'index': 'Clase'}))}

#### Comparación con referencias sencillas

{tabla(referencias.round(2))}

La diferencia de la red frente a la logística es de **{diferencia:.2f} puntos porcentuales** en esta ejecución.
Se utilizan las mismas cinco entradas y la misma prueba. No demuestra superioridad general ni significación estadística.

#### Ejemplos reales

{tabla(ejemplos)}

Son dos aciertos y dos errores, escogidos para explicar el resultado. El modelo no recibe los títulos y estos ejemplos no se usan para elegirlo.
''')

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(historial['Época'], historial['Loss Train'], '--', color='tab:red', label='Train Loss')
    ax1.plot(historial['Época'], historial['Loss Val'], color='tab:orange', label='Val Loss')
    ax1.set(xlabel='Época', ylabel='Loss', title='Curva del Modelo Baseline')
    ax1.legend(loc='upper right')
    ax2 = ax1.twinx()
    ax2.plot(historial['Época'], historial['Accuracy Val (%)'], color='tab:blue')
    ax2.set_ylabel('Accuracy de validación (%)')
    fig.tight_layout()
    fig.savefig(GRAFICOS / 'baseline.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    metricas.loc[clases, ['precision', 'recall', 'f1-score']].plot.bar(ax=ax)
    ax.set(title='Métricas por clase del modelo inicial', ylabel='Puntuación', ylim=(0, 1))
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    fig.savefig(GRAFICOS / 'metricas_baseline.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(comparacion['ID'], comparacion['Val Accuracy (%)'], yerr=comparacion['Desviación accuracy'], capsize=4, color='#287c8e')
    axes[0].set(title='Accuracy media de validación', ylabel='Accuracy (%)')
    comparacion.set_index('ID')[['Train Loss Final', 'Val Loss Final']].plot.bar(ax=axes[1])
    axes[1].set(title='Pérdida final media', ylabel='Loss')
    axes[1].tick_params(axis='x', rotation=0)
    fig.tight_layout()
    fig.savefig(GRAFICOS / 'comparacion.png', dpi=150)
    plt.close(fig)

    matriz = pd.read_csv(RESULTADOS / 'matriz_confusion.csv', index_col=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    imagen = ax.imshow(matriz, cmap='Blues')
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, str(matriz.iloc[i, j]), ha='center', va='center', color='white' if matriz.iloc[i, j] > matriz.to_numpy().max() / 2 else 'black')
    ax.set_xticks(range(len(clases)), clases, rotation=25, ha='right')
    ax.set_yticks(range(len(clases)), clases)
    ax.set(xlabel='Predicción', ylabel='Categoría real', title='Prueba final · Semilla 42')
    fig.colorbar(imagen, ax=ax)
    fig.tight_layout()
    fig.savefig(GRAFICOS / 'matriz_confusion.png', dpi=150)
    plt.close(fig)
    ganador = resumen['experimento_elegido']
    curva = pd.read_csv(RESULTADOS / f'historial_{ganador}_42.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curva['Época'], curva['Loss Train'], label='Entrenamiento')
    ax.plot(curva['Época'], curva['Loss Val'], label='Validación')
    ax.set(title=f'Curvas de {ganador} · Semilla 42', xlabel='Época', ylabel='Loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / 'curvas_ganador.png', dpi=150)
    plt.close(fig)
    print('Tablas y cinco figuras del entrenamiento generadas desde los resultados.')


if __name__ == '__main__':
    generar()
