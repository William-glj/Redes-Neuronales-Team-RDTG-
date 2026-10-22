# Entender la parte de PyTorch antes de exponer

Tu objetivo es poder explicar qué entra, qué sale, cómo aprende y cómo
comprobamos el resultado. Practica primero los apartados 1–5; después, las preguntas.

## 1. Una fila es una película

Cada película tiene cinco números: presupuesto, duración, votos, nota y año.
Antes de llegar a la red, presupuesto y votos pasan por `log1p` y las cinco
columnas se normalizan. El título ayuda a leer ejemplos, pero no es una entrada.

`X` contiene las entradas. `y` contiene las respuestas conocidas:
0 = Baja, 1 = Media, 2 = Media-alta y 3 = Alta.
La respuesta nunca se incluye entre los cinco números de `X`.

Un tensor es una estructura de números parecida a un array de NumPy que
PyTorch utiliza para hacer cálculos y obtener gradientes.

## 2. Para qué sirven las tres partes

| Conjunto | Películas | Qué hacemos con ellas |
| --- | --- | --- |
| Entrenamiento | 893 | Aprender los pesos y calcular el preprocesado. |
| Validación | 191 | Comparar configuraciones y guardar una época. |
| Prueba | 192 | Medir al final la configuración elegida. |

Si elegimos una red mirando la prueba, estaríamos aprovechando información
de la evaluación final. `stratify` mantiene proporciones similares de clases.
La semilla de la separación es siempre 42.

Normalizar consiste en restar la media de entrenamiento y dividir por su
desviación. Aplicamos esos mismos valores a validación y prueba. No calculamos
una media distinta para cada conjunto.

## 3. Leer la arquitectura

```python
modelo = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 4)
)
```

`Sequential` aplica las operaciones en ese orden. `Linear(5, 64)` recibe cinco
valores y calcula 64 resultados usando pesos y sesgos aprendidos. `ReLU`
deja los positivos como están y convierte los negativos en cero.

| Paso | Forma de un lote completo de 32 películas |
| --- | --- |
| Entrada | 32 filas × 5 variables |
| Primera Linear y ReLU | 32 × 64 |
| Segunda Linear y ReLU | 32 × 32 |
| Salida | 32 × 4 puntuaciones |

La última tanda puede tener menos de 32 películas. Cada fila sigue siendo
una película durante todo el recorrido; las neuronas cambian las columnas.

Las cuatro puntuaciones se llaman logits. Por ejemplo, `[0.2, -0.5, 0.8, 1.4]`
es un ejemplo inventado para entender el formato: `argmax(1)` escogería el
índice 3, Alta. No son porcentajes ni tienen que sumar 1.

La red tiene 2.596 parámetros: `(5×64 + 64) + (64×32 + 32) + (32×4 + 4)`.
Los términos que se suman al producto son los sesgos.

## 4. Leer el bucle de aprendizaje

```python
optimizador.zero_grad()
salida = modelo(x_lote)
error = perdida(salida, y_lote)
error.backward()
optimizador.step()
```

1. `zero_grad()` limpia los gradientes anteriores: PyTorch los acumula si no lo hacemos.
2. `modelo(x_lote)` calcula cuatro puntuaciones para cada película.
3. `CrossEntropyLoss` compara esas puntuaciones con las etiquetas correctas.
4. `backward()` calcula cómo influye cada parámetro en el error.
5. Adam usa esos gradientes para actualizar los parámetros.

No añadimos Softmax antes de CrossEntropyLoss, porque esta pérdida ya incorpora
la operación necesaria. Para elegir la clase basta con la puntuación más alta.

Una época recorre todas las películas de entrenamiento; un lote es un grupo
utilizado para una actualización. El learning rate controla el tamaño de
los ajustes. Cambiar el lote no cambia el número de variables ni de clases.

## 5. Elegir y evaluar son dos pasos distintos

En cada entrenamiento guardamos los pesos de la menor pérdida de validación,
que pueden ser de una época anterior a la 15. Después comparamos las ocho
configuraciones por su F1 macro medio de validación entre tres semillas.
La elegida es Exp_2. Solo entonces se evalúa en prueba.

Las tres semillas cambian la inicialización y el orden de los lotes.
No cambian qué películas pertenecen a cada conjunto: no es validación cruzada.

`modelo.eval()` activa el modo evaluación. `torch.no_grad()` evita calcular
gradientes cuando solo queremos medir. En estas redes, que no tienen Dropout
ni BatchNorm, `eval()` no cambia las capas, pero deja claro el uso del modelo.

## 6. Las cifras que puedes decir

- Exp_2: cinco entradas, capas ocultas 64 y 32, cuatro salidas.
- Batch 32, Adam, learning rate 0,01, hasta 15 épocas.
- F1 macro medio de validación: 0,633.
- Accuracy media de prueba: 60,9 %; F1 macro medio: 0,603.
- Semilla 42 de la matriz: 117 aciertos de 192. La media de las tres semillas
  también redondea a 60,9 %, pero son cálculos distintos.

No digas «63,3 % de aciertos» al leer F1 = 0,633. La accuracy media de validación
es otra columna: aproximadamente 63,7 %.

## 7. Si abren el código completo

En `preparacion_hiperparametros_modulares.py`, la clase `PeliculasNeuralNetwork`
construye la misma idea con una lista. El bucle añade una `Linear` y una `ReLU`
por cada tamaño de capa oculta. Después añade la salida de cuatro neuronas.
`nn.Sequential(*layers)` pasa los elementos de la lista como argumentos
separados. `forward` aplica esa secuencia a las entradas.

`copy.deepcopy(modelo.state_dict())` conserva una copia de los mejores pesos.
Si solo guardásemos una referencia al modelo que sigue aprendiendo, perderíamos
el estado de esa época. `torch.save` guarda los pesos para volver a cargarlos.

`preparacion_datos_entrenamiento.py` contiene el modelo inicial de 64–64 y
diez épocas. No lo confundas con la configuración seleccionada de 64–32.

## 8. Comprobación oral sin mirar

Responde en voz alta:

1. ¿Por qué hay cinco entradas y cuatro salidas?
2. ¿Qué ocurriría si usáramos `popularity` entre las entradas?
3. ¿Para qué necesitamos validación además de prueba?
4. ¿Qué hace cada una de las cinco líneas del bucle?
5. ¿Por qué tres semillas no son tres particiones?
6. ¿Por qué F1 y accuracy no significan lo mismo?

Si puedes explicarlo con tus palabras, conoces lo esencial de tu intervención.
