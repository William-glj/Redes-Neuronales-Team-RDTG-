# Guía de exposición del grupo

Ensayar unos 8–10 minutos, ajustándolo al tiempo que dé el profesor. Todos hablan.
Abrir `presentacion.html` antes de empezar. No hace falta entrenar en directo.

| Persona | Parte | Qué mostrar |
| --- | --- | --- |
| José Ramón | Introducción y problema | Qué es clasificación; categoría y cinco entradas. |
| Guillermo | Fase A | Limpieza, nulos, IQR y dos gráficos. |
| Dylan | Fase B | Separación, StandardScaler, modelo inicial y cómo aprende. |
| Tomás | Fase C | Siete configuraciones, tres semillas, tablas y prueba final. |

## Introducción

Clasificamos películas en Low, Medium, Moderately High y High. Es aprendizaje supervisado:
conocemos las etiquetas al entrenar. Usamos presupuesto, duración, votos, nota media y año.
Popularity no entra porque determina directamente la categoría. Votos y nota media son
posteriores al estreno; no prometemos predecir el éxito antes de estrenar.

## Fase A

El CSV recibido tiene 1.287 películas. Conservamos el filtro comercial del equipo y excluimos
una etiqueta ausente: quedan 1.276. Corregimos doce fechas con el siglo incorrecto.
Pandas trabaja con tablas, NumPy con cálculos y Matplotlib con los gráficos.

El IQR señala extremos. Una película cara puede ser válida: no borramos automáticamente
los atípicos. Los gráficos muestran presupuestos asimétricos, categorías parecidas y duraciones
que se solapan. Una correlación no demuestra una causa.

## Fase B

Hay 893 películas para aprender, 191 para elegir y 192 para evaluar al final. No comparten IDs.
StandardScaler aprende medias y desviaciones solo con entrenamiento. No limita todo a 0–1.

Se mantiene el modelo inicial: 5 entradas, dos capas de 64 y cuatro salidas. Diez épocas,
Adam, tasa 0,01 y lotes de 32. Su accuracy de validación es 62,30 %.

Una neurona combina entradas con pesos y un sesgo. ReLU introduce no linealidad.
Las salidas son logits; argmax escoge la mayor. En esta versión LabelEncoder ordena:
High=0, Low=1, Medium=2, Moderately High=3.

- `zero_grad`: limpia gradientes anteriores.
- `modelo`: calcula puntuaciones.
- `CrossEntropyLoss`: mide el error respecto a la etiqueta.
- `backward`: calcula los gradientes.
- `step`: actualiza los pesos con Adam.

Una época recorre entrenamiento. Un lote es el grupo de películas de cada actualización.
No actualizamos pesos usando validación ni prueba.

## Fase C

Conservamos las siete configuraciones originales. Cada una se repite con tres semillas:
21 entrenamientos de quince épocas. La semilla controla pesos iniciales y orden de los lotes;
las películas de cada conjunto siguen siendo las mismas. No es validación cruzada.

Exp_2/4/5 compara lotes; Exp_2/6/7 compara tasas. Exp_1/2/3 cambia profundidad y anchura
juntas, así que no permite atribuir el resultado a una sola de ellas.

Se selecciona por accuracy media de validación al terminar quince épocas; desempate por
menor pérdida media. Gana **Exp_1: 5 → 32 → 4**, lote 32, tasa 0,01 y 324 parámetros.
Su media de validación es **61,78 %**. No se escoge por el resultado de prueba.

En prueba obtiene **67,53 % de accuracy medio** y **F1 macro 0,673**. La regresión logística
alcanza **63,54 %**: la diferencia es aproximadamente cuatro puntos en esta prueba.
La matriz corresponde a la semilla 42: **129 aciertos de 192**, 67,19 %.

Accuracy cuenta aciertos. F1 macro promedia el F1 de cada categoría con el mismo peso.
En la matriz, filas son reales y columnas predicciones. La diagonal son los aciertos.

## Preguntas que todos deben poder responder

- ¿Por qué tres conjuntos? Aprender, elegir y evaluar tienen funciones diferentes.
- ¿Por qué una red? Exploramos relaciones no lineales y la comparamos con un modelo sencillo.
- ¿Por qué cambia el resultado antiguo? Cambiaron la separación, la limpieza y la repetición de los entrenamientos.
- ¿Por qué prueba supera validación? Son películas distintas; no tiene que salir menor. No utilizamos prueba para escoger.
- ¿Por qué no 100 %? Las cinco entradas no explican toda la popularidad y hay clases que se solapan.
- ¿Más capas es mejor? Los resultados no lo garantizan; en esta rejilla gana una capa de 32.

La muestra es pequeña y seleccionada. Tres semillas sobre una división no prueban superioridad
general ni significación estadística. Si preguntan una causa no investigada, explicar lo observado
sin inventarla. Revisar también el informe y los CSV para poder señalar ejemplos reales.
