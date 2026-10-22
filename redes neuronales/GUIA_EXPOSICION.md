# Guía para preparar la defensa

No es un texto que haya que memorizar literalmente. Repartidlo, ejecutad el
código y explicadlo con vuestras palabras. Los resultados concretos deben
consultarse DESPUÉS de entrenar; no hay un porcentaje prometido.

## Reparto sugerido: unos 8–10 minutos en total

| Persona | Parte | Tiempo orientativo |
| --- | --- | --- |
| José Ramón | Tema, datos y problema de clasificación. | 2 minutos |
| Guillermo | Limpieza, nulos, atípicos y gráficos. | 2–3 minutos |
| Dylan | Preparación de datos y red PyTorch. | 2–3 minutos |
| Tomás | Experimentos, resultados y limitaciones. | 2–3 minutos |

Adaptad el tiempo a lo que haya pedido el profesor. Los cuatro debéis saber
qué se clasifica y por qué no utilizamos `popularity` como entrada.

## José Ramón: qué hacemos y con qué datos

- Tema: cine. Clasificamos películas en cuatro niveles de popularidad.
- El archivo recibido tiene 1.287 películas; es una selección histórica.
- Confirmar y citar la fuente exacta antes de la defensa. No inventar que se
  descargó de Kaggle si no se conserva ese enlace.
- Explicar que es clasificación: la respuesta es una categoría, no una cifra.
- Es un análisis posterior al estreno, porque tenemos recaudación y votos.
- No identificamos popularidad con calidad cinematográfica ni con beneficio.

Mostrar en el informe la tabla de categorías y la descripción del problema.

## Guillermo: cómo preparamos una base fiable

- Conservamos el CSV recibido y trabajamos sobre una copia.
- Quitamos el índice exportado y comprobamos duplicados por identificador.
- Encontramos una película sin categoría y la eliminamos: quedan 1.276.
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

Tras entrenar, completar estas notas personales usando las tablas:

1. Configuración elegida y su F1 medio de validación.
2. Accuracy y F1 de prueba: distinguir el promedio de tres semillas de la
   evaluación de la semilla 42 que aparece en la matriz.
3. Error más frecuente de la matriz y dos películas mal clasificadas, obtenidas
   de `resultados/predicciones_prueba.csv`.
4. Comparación con la clase mayoritaria y la regresión logística.
5. Qué ocurre al aumentar la complejidad: mejor rendimiento, empate o sobreajuste.
   Describir lo que realmente muestran los resultados.

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
