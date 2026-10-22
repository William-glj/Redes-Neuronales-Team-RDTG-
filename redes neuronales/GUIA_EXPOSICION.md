# Guía para preparar la defensa

Usad el informe abierto y explicadlo con vuestras palabras. Las cifras de esta
guía son las de la ejecución comprobada: no son porcentajes prometidos para
otros datos. No hace falta entrenar durante la exposición.

## Las ocho cifras que debéis compartir

| Dato | Valor y significado |
| --- | --- |
| Muestra recibida / final | 1.287 / 1.276 películas; se excluyen 10 por el umbral y 1 sin etiqueta. |
| Entrenamiento / validación / prueba | 893 / 191 / 192; son conjuntos distintos. |
| Entradas / salidas | 5 variables / 4 categorías. |
| Configuraciones / repeticiones | 8 configuraciones × 3 semillas = 24 entrenamientos, más el inicial. |
| Configuración elegida | Exp_2: 64 y 32 neuronas, batch 32, learning rate 0,01. |
| Criterio de elección | F1 macro medio de validación: 0,633. |
| Resultado en prueba | Accuracy media 60,9 %; F1 macro medio 0,603. |
| Referencia sencilla | Regresión logística: 58,9 % de accuracy en prueba. |

La matriz usa la semilla 42: 117 aciertos de 192 películas. Su accuracy
también redondea a 60,9 %, pero no es el mismo cálculo que el promedio de semillas.

## Reparto sugerido: unos 8–10 minutos en total

| Persona | Parte | Tiempo orientativo |
| --- | --- | --- |
| José Ramón | Tema, datos y problema de clasificación. | 1,5–2 minutos |
| Guillermo | Limpieza, nulos, atípicos y gráficos. | 2 minutos |
| Dylan | Preparación de datos y red PyTorch. | 2–2,5 minutos |
| Tomás | Experimentos, resultados y limitaciones. | 2–2,5 minutos |

Adaptad el tiempo a lo que haya pedido el profesor. Los cuatro debéis saber
qué se clasifica y por qué no utilizamos `popularity` como entrada.

## José Ramón: qué hacemos y con qué datos

- Tema: cine. Clasificamos películas en cuatro niveles de popularidad.
- El archivo recibido tiene 1.287 películas; es una selección histórica.
- Fuente pública contrastada: TMDb Movies Dataset de Juzer Shakir en Kaggle.
  El público tiene 10.866 filas; nuestra copia es una selección de 1.287.
  `comprobar_fuente.py` verifica las coincidencias y reproduce las etiquetas.
  No afirmamos saber el enlace usado para descargar la copia antigua.
- Explicar que es clasificación: la respuesta es una categoría, no una cifra.
- Es un análisis posterior al estreno: la red utiliza votos y valoraciones.
- No identificamos popularidad con calidad cinematográfica ni con beneficio.

Mostrar en el informe la tabla de categorías y la descripción del problema.

## Guillermo: cómo preparamos una base fiable

- Conservamos el CSV recibido y trabajamos sobre una copia.
- Quitamos el índice exportado y comprobamos duplicados por identificador.
- De 1.287, el umbral comercial excluye 10 y la etiqueta ausente excluye 1:
  quedan 1.276. No usamos una quinta clase llamada `nan`.
- Corregimos 12 fechas con el siglo incorrecto contrastando el año con la
  fuente pública. `release_year`, que sí entra en la red, ya era correcto.
- Conservamos el umbral comercial de 10.000 del equipo. Es una selección de muestra, no una prueba de que las películas excluidas estén mal.
- El IQR señala valores extremos; no demuestra que estén mal.
- Aplicamos logaritmos a presupuesto y votos para reducir escalas.
- Si faltan entradas numéricas, usamos medianas de entrenamiento. En la muestra
  actual no han hecho falta esas imputaciones.
- Enseñar el reparto equilibrado y dos gráficos más. Explicar lo que se ve,
  no limitarse a decir el nombre de cada gráfico.

Pregunta útil: ¿por qué hay más apariciones de géneros que películas?
Porque una película puede ser a la vez drama, acción y aventura.

## Dylan: cómo aprende la red

Tenemos cinco entradas, las mismas que en desarrollo: presupuesto, duración,
votos, nota media y año. Los géneros se estudian en el EDA, pero no entran a la red.

Separación: 893 películas de entrenamiento, 191 de validación y 192 de prueba.
Equivale aproximadamente a 70 %, 15 % y 15 %.
Los conjuntos no comparten películas. La media y la desviación se calculan
solo con entrenamiento y se aplican a los tres conjuntos.

Explicar este ejemplo con dos capas ocultas:

```python
nn.Linear(5, 64)  # cinco entradas, 64 neuronas
nn.ReLU()
nn.Linear(64, 32)  # segunda capa oculta
nn.ReLU()
nn.Linear(32, 4)   # cuatro puntuaciones, una por clase
```

- `nn` es el módulo de redes neuronales de PyTorch.
- Un tensor es una estructura de números, similar a un array, con operaciones
  que PyTorch puede usar para calcular gradientes.
- Los pesos y sesgos son los valores que aprende la red.
- ReLU introduce relaciones no lineales; sin activaciones, apilar estas capas
  lineales seguiría siendo una transformación lineal.
- CrossEntropyLoss compara las cuatro puntuaciones con la categoría real.
- Adam actualiza los pesos usando los gradientes.
- `argmax(1)` escoge la salida con mayor puntuación para cada película.
- `eval()` pone la red en modo evaluación; `no_grad()` evita calcular gradientes
  cuando solo queremos medir resultados.

Para comprender la clase modular `PeliculasNeuralNetwork`, seguir el bucle
de `__init__`: empieza con cinco entradas, añade una capa Linear y una ReLU
por cada tamaño de la lista y termina con cuatro salidas. El asterisco de
`nn.Sequential(*layers)` pasa los elementos de la lista como argumentos
separados. `forward` aplica esa secuencia. `construir_red` crea una instancia
de esa clase con cuatro salidas.

## Tomás: qué hemos probado y aprendido

- Ocho configuraciones, tres semillas por configuración: 24 entrenamientos,
  más el modelo inicial de dos capas de 64 durante diez épocas.
- Exp_1/Exp_8: cambia la anchura. Exp_8/Exp_2: se añade una capa.
- Exp_2/Exp_3: cambian anchura y profundidad a la vez; no aislamos ambos efectos.
- Exp_2/Exp_4/Exp_5: cambia el batch. Exp_2/Exp_6/Exp_7: cambia el learning rate.
- Elegimos el mayor F1 macro MEDIO de validación, no el mayor de prueba.
- En cada entrenamiento recuperamos la época de menor pérdida de validación.
- La semilla 42 se fija para la matriz de confusión; no buscamos en prueba
  qué semilla queda mejor.

Conclusiones concretas para explicar:

1. Exp_2 gana por F1 medio de validación: 0,633. Su prueba da 60,9 % de aciertos
   y F1 medio 0,603. El descenso frente a validación recuerda que elegir una
   configuración y evaluarla al final son tareas distintas.
2. La clase mayoritaria acierta un 25 % en prueba; la regresión logística,
   un 58,9 %. La diferencia de la red frente a la logística es pequeña:
   unos 2,1 puntos porcentuales de accuracy, no una mejora demostrada en general.
3. La red más grande, Exp_3, obtiene F1 de validación 0,594: más tamaño no
   asegura mejorar. Exp_7, learning rate 0,0001, queda en 0,373 con 15 épocas;
   sería incorrecto asegurar que esa tasa nunca funciona.
4. En la matriz, filas = clase real y columnas = predicción. La diagonal suma
   117. El error más frecuente es Media → Baja: 15 películas.
5. Shrek 2 y The Incredibles son aciertos de clase Alta. Gangster Squad es
   Media-alta y se predice Alta; Boys Don't Cry es Media-alta y se predice Baja.
   Los ejemplos salen del CSV de prueba, no se han inventado.

## Guion breve para ensayar las cuatro intervenciones

No lo leáis palabra por palabra. Usadlo para practicar con el informe delante.

**José Ramón — Introducción y datos.**

«Nuestro tema es el cine y hemos planteado una clasificación de películas en
cuatro niveles de popularidad. La respuesta es una categoría: baja, media,
media-alta o alta. No buscamos calcular una recaudación. Trabajamos con una
selección histórica de 1.287 películas. La hemos contrastado con el dataset
público de TMDb de Juzer Shakir en Kaggle. Las categorías recibidas se pueden
reproducir dividiendo la popularidad de la muestra por cuartiles. No son
etiquetas oficiales de TMDb. Como usamos votos y valoraciones, el análisis
es posterior al estreno. Para estudiar cómo se comporta una red, probamos
varias configuraciones y la comparamos con un clasificador sencillo.
Guillermo explica ahora cómo limpiamos los datos.»

Mostrar: introducción, fuente y tabla de las cuatro etiquetas.

**Guillermo — EDA y limpieza.**

«Conservamos el archivo recibido y limpiamos una copia. Comprobamos nulos y
duplicados. Mantenemos el filtro de nuestro proyecto: presupuesto y recaudación
superiores a 10.000. Esto descarta diez películas y limita la muestra; no
significa que una película barata esté mal. Quitamos además una película sin
categoría y quedan 1.276. La comparación con la fuente detectó doce fechas
con el siglo equivocado, que corregimos usando el año contrastado. Para los
atípicos usamos el criterio IQR. Conservamos los extremos válidos: una película
con muchos votos no es un error por tener muchos. Este histograma muestra
por qué aplicamos logaritmo al presupuesto. Las clases tienen tamaños
parecidos y las duraciones se solapan. Dylan explica cómo pasan los datos
limpios a la red.»

Mostrar: resumen de limpieza, histograma del presupuesto y reparto de clases.

**Dylan — Separación y entrenamiento.**

«La red recibe cinco variables: presupuesto, duración, votos, nota media y año.
Excluimos popularity porque está ligada a la respuesta. Separamos 893 películas
para entrenamiento, 191 para validación y 192 para prueba. Con entrenamiento
aprendemos los pesos; con validación elegimos la configuración; la prueba
queda para evaluar al final. Aplicamos logaritmo a presupuesto y votos y
normalizamos con estadísticas calculadas solo en entrenamiento. En el ejemplo,
Linear pasa de cinco entradas a 64 neuronas, después a 32 y al final a cuatro
salidas. ReLU permite relaciones no lineales. Las salidas son puntuaciones:
elegimos la más alta. Durante cada lote calculamos el error con CrossEntropyLoss,
backward calcula gradientes y Adam actualiza los pesos. Una época recorre todo
entrenamiento. Tomás compara los experimentos.»

Mostrar: separación, cinco entradas, red `5 → 64 → 32 → 4` y bucle de aprendizaje.
Para entenderlo despacio: `EXPLICACION_DYLAN.md`.

**Tomás — Comparación y conclusión.**

«Probamos ocho configuraciones, cada una con tres semillas: 24 entrenamientos,
además del modelo inicial. Cambiamos capas, neuronas, tamaño del lote y tasa de
aprendizaje. Exp_2 tiene el mayor F1 macro medio de validación: 0,633. Por eso
elegimos 64 y 32 neuronas, lote 32 y tasa 0,01. En prueba obtiene un 60,9 % de
aciertos de media; la logística obtiene un 58,9 %. La diferencia es pequeña y
no demuestra que una red siempre sea mejor. Aquí la diagonal de la matriz
contiene los aciertos. El error más frecuente es confundir Media con Baja.
El modelo acierta Shrek 2 como Alta y se equivoca con Gangster Squad, que es
Media-alta y predice Alta. Una red más grande tampoco mejoró: los datos y el
entrenamiento importan. Mejoraríamos la evaluación con otras particiones y
datos más recientes.»

Mostrar: tabla de experimentos, curva, matriz y ejemplos. Terminar con las
limitaciones, sin prometer predecir futuros éxitos.

## Preguntas que os pueden hacer a cualquiera

**¿Por qué no usáis popularity como entrada?**
Porque la categoría está ligada a esa puntuación. Sería dar la respuesta a
la red y producir una evaluación engañosa.

**¿Por qué sí usáis votos?**
Porque hemos definido una clasificación posterior al estreno. Es una señal
relacionada con el propio indicador; no presentamos el ejercicio como una
predicción independiente de éxito futuro. Es una limitación que reconocemos.

**¿Por qué no basta con dividir entrenamiento y prueba?**
Porque necesitamos elegir la configuración sin utilizar la prueba. Para eso
reservamos validación.

**¿Qué es una época? ¿Y un lote?**
Una época recorre el conjunto de entrenamiento. Un lote es el grupo de ejemplos
que se usa en una actualización de pesos.

**¿Qué es la tasa de aprendizaje?**
Controla cuánto se ajustan los pesos en cada actualización. Una tasa muy baja
puede aprender despacio; una alta puede dificultar una convergencia estable.

**¿Qué significa sobreajuste?**
Que el modelo mejora en entrenamiento pero no generaliza igual a datos nuevos.
Puede verse cuando baja la pérdida de entrenamiento y empeora validación.

**¿Qué es accuracy? ¿Qué es F1 macro?**
Accuracy: proporción de aciertos. F1 combina precisión y exhaustividad. Macro
calcula el F1 de cada clase y hace la media dándoles el mismo peso.

**¿Por qué tres semillas? ¿Es validación cruzada?**
Para comprobar la sensibilidad a la inicialización y al orden de los lotes.
No es validación cruzada: las películas de cada conjunto no cambian.

**¿Cómo sabemos que una red es necesaria?**
No lo sabemos por adelantado. Probamos relaciones no lineales y varias
arquitecturas, y comparamos con una referencia sencilla. Si la red no supera
la regresión logística, lo decimos. No se debe prometer que una red siempre gana.

**¿Qué hace Quarto?**
Convierte el documento con explicaciones, código, tablas y figuras en una web.
El entrenamiento lo hacen los scripts de Python, no el navegador del visitante.

## Ensayo final

Abrir el informe publicado y una copia local. Cada uno debe practicar su parte
con los gráficos delante, sin leer párrafos completos. Preparad una transición
a la siguiente persona. No ejecutéis los 24 entrenamientos durante la exposición:
mostrad las métricas guardadas y explicad cómo se reproducen.
