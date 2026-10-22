# Entender el proyecto desde cero

Guía para José Ramón, Guillermo, Dylan y Tomás. Todas las cifras proceden
de la ejecución comprobada del proyecto. Los ejemplos inventados para explicar
un concepto se indican expresamente.

## 1. Qué estamos intentando hacer

Queremos clasificar películas en cuatro niveles de popularidad: Baja, Media,
Media-alta y Alta. La red recibe características de cada película y devuelve
una de esas categorías. No recomienda películas a una persona ni calcula
cuánto dinero recaudará una película.

Es aprendizaje supervisado: durante el entrenamiento tenemos ejemplos con
sus respuestas conocidas. La red utiliza esas respuestas para ajustar sus
parámetros. Para comprobar si aprendió algo útil, la evaluamos con películas
que no utilizó para actualizar sus parámetros.

El análisis es posterior al estreno: usamos número de votos y nota media,
información que se obtiene cuando la gente ya ha visto la película.

## 2. Qué significa clasificación

Clasificación significa escoger una categoría. Una regresión numérica buscaría
una cantidad, como euros de recaudación. En nuestro proyecto la respuesta es
una etiqueta. Que la codifiquemos con un número del 0 al 3 no la convierte
en una regresión: esos números identifican clases.

| Concepto | Nuestro proyecto |
| --- | --- |
| Una observación | Una película. |
| Variables explicativas, X | Cinco características de la película. |
| Variable dependiente, y | Su categoría de popularidad. |
| Modelo | Una función con parámetros aprendidos. |
| Predicción | La categoría que elige esa función. |

## 3. De dónde salen los datos

El CSV recibido tiene 1.287 películas, de 1961 a 2015, y 24 columnas contando
el índice exportado. Hemos contrastado una fuente pública equivalente:
TMDb Movies Dataset, publicado por Juzer Shakir en Kaggle.
https://www.kaggle.com/datasets/juzershakir/tmdb-movies-dataset

El público tiene 10.866 películas y 21 columnas. Los mismos 1.287 identificadores
se obtienen sustituyendo por nulos los ceros de presupuesto y recaudación,
eliminando filas con nulos y quitando duplicados. Reproducir esa selección
no demuestra desde qué enlace se descargó la copia antigua.

`comprobar_fuente.py` contrasta las columnas y documenta diferencias de
redondeo, formato y doce fechas incorrectas. También comprueba las dos
columnas añadidas, `profit` y `popularity_level`. La ficha indica licencia
Unknown: citamos el dataset sin inventar otra licencia.

Trabajamos con la selección recibida, no con las 10.866 películas completas.
La selección previa puede favorecer películas con información más completa.

## 4. Popularity y popularity_level

`popularity` es la puntuación numérica de popularidad que viene en el CSV.
`vote_average` es otra columna: contiene la nota media del público.
Una película puede ser muy conocida y recibir una nota baja.

`popularity_level` agrupa la puntuación en cuatro categorías. Hemos comprobado
que las etiquetas recibidas se reproducen con cuartiles de las 1.287 películas.
Los cuartiles marcan los límites que separan aproximadamente cuatro grupos
del mismo tamaño. Estos cortes son propios de la muestra, no categorías
oficiales de TMDB.

Ejemplos numéricos ilustrativos, usando nuestros cortes:

| Popularity | Categoría |
| --- | --- |
| 0,5 | Baja |
| 1 | Media |
| 1,5 | Media-alta |
| 3 | Alta |

Excluimos `popularity` de las entradas porque determina directamente la
etiqueta. Si se la diéramos a la red, bastaría con aprender esos cortes.
Sería una fuga de información respecto a nuestro objetivo de clasificar
a partir de otras características.

Los votos también están relacionados con el indicador: reconocemos esa
limitación y no presentamos el ejercicio como una predicción independiente
del éxito futuro.

## 5. Qué hace el EDA

EDA significa análisis exploratorio de datos: comprender qué hay en el CSV,
buscar problemas y describir sus distribuciones antes de entrenar.

Pandas organiza los datos en una tabla llamada DataFrame. NumPy permite
operaciones numéricas con arrays. Matplotlib dibuja las figuras.

```python
df = pd.read_csv("datos/peliculas_original.csv")
print(df.shape)
print(df.isna().sum())
```

`shape` indica filas y columnas. `isna()` detecta valores ausentes y `sum()`
cuenta los de cada columna. Un nulo es un dato desconocido; no equivale a cero.

Nuestra limpieza hace lo siguiente:

1. Conserva el CSV recibido y trabaja sobre una copia.
2. Quita el índice exportado, que no es una característica de la película.
3. Comprueba duplicados por `id`: en el recibido hay cero.
4. Mantiene el filtro comercial del equipo: presupuesto y recaudación
   superiores a 10.000. Se excluyen diez películas. Es una selección de
   muestra, no una demostración de que todos los importes bajos sean errores.
5. Elimina una película sin categoría. Convertir el nulo a texto crearía
   una quinta clase incorrecta llamada `nan`.
6. Corrige doce fechas con el siglo incorrecto, contrastando el año con la
   fuente pública. La columna `release_year` ya era correcta.
7. Revisa números imposibles y valores no finitos. Si faltasen entradas
   numéricas, su mediana se calcularía después con entrenamiento. En esta
   muestra no ha sido necesario imputar esas entradas.

Resultado: **1.287 − 10 − 1 = 1.276 películas**.

Las doce fechas se corrigen en el CSV limpio. No se cambia el CSV recibido
ni las cinco entradas que ya utilizaba el modelo.

## 6. Qué hacemos con los valores atípicos

Un valor atípico está alejado de la mayoría. No tiene por qué ser incorrecto:
una superproducción puede tener un presupuesto enorme y válido.

Usamos el IQR, distancia entre el primer y el tercer cuartil. Se señalan
valores inferiores a Q1 − 1,5 × IQR o superiores a Q3 + 1,5 × IQR.
Conservamos estos extremos y reducimos la diferencia de escala mediante
logaritmos en presupuesto y votos. No eliminamos automáticamente todos
los puntos que aparecen fuera de los bigotes de un boxplot.

## 7. Qué enseñan los gráficos del EDA

| Gráfico | Qué debes explicar |
| --- | --- |
| Categorías | Los cuatro grupos tienen tamaños parecidos: 314, 319, 321 y 322. |
| Presupuesto original y transformado | Hay diferencias enormes de escala; el logaritmo las comprime. |
| Duración por categoría | Las distribuciones se solapan; la duración por sí sola no separa las clases. |
| Géneros | Una película puede tener varios. Las apariciones sumadas pueden superar las películas. |
| Correlaciones | Resumen de asociaciones lineales; una correlación no demuestra causalidad. |

`genres` se separa por `|` para contar géneros. Eso se utiliza en el gráfico,
pero no añade variables de género a la red de cinco entradas.

## 8. Qué entra y qué sale de la red

| Entrada | Significado | Preparación |
| --- | --- | --- |
| budget | Presupuesto | Logaritmo y normalización. |
| runtime | Duración | Normalización. |
| vote_count | Número de votos | Logaritmo y normalización. |
| vote_average | Nota media | Normalización. |
| release_year | Año de estreno | Normalización. |

`revenue`, los géneros y `popularity` aparecen en el EDA, pero no son entradas.
Tampoco usamos título, identificadores ni URLs.

La respuesta conocida se codifica así: Low = 0, Medium = 1,
Moderately High = 2 y High = 3. Las cuatro salidas de la red corresponden
siempre a ese mismo orden.

## 9. Por qué dividimos los datos

| Conjunto | Películas | Función |
| --- | --- | --- |
| Entrenamiento | 893 | Aprender pesos y calcular el preprocesado. |
| Validación | 191 | Elegir configuración y época. |
| Prueba | 192 | Evaluar al final la configuración elegida. |

Es aproximadamente 70 %, 15 % y 15 %. `stratify` conserva proporciones
parecidas de clases. La semilla fija 42 permite repetir la división.
Los conjuntos no comparten identificadores.

Piensa en ejercicios para aprender, un simulacro para decidir qué método
funciona y un examen final. Si escoges el método mirando repetidamente
el examen final, su resultado deja de ser una evaluación independiente.

Las etiquetas ya venían definidas antes de este experimento. No las
redefinimos para obtener mejor nota en prueba.

## 10. Logaritmos, medianas y normalización

La red no recibe directamente un presupuesto de cientos de millones junto
a una nota de unos pocos puntos. Preparar escalas facilita el entrenamiento.

`np.log1p(x)` calcula log(1 + x). Comprime los valores grandes y está definido
cuando x es cero. Lo aplicamos a presupuesto y número de votos.

Si una entrada numérica estuviera ausente, se rellenaría con la mediana de
esa columna en entrenamiento. La mediana es el valor central al ordenar.

Después usamos StandardScaler: restar la media de entrenamiento y dividir
por su desviación típica. Las columnas de entrenamiento quedan aproximadamente
con media cero y desviación uno. Esto no limita los valores al intervalo 0–1.

```python
scaler = StandardScaler()
x_entreno = scaler.fit_transform(valores_entreno)
x_val = scaler.transform(valores_val)
x_prueba = scaler.transform(valores_prueba)
```

`fit_transform` aprende el escalado y lo aplica. `transform` aplica el escalado
ya aprendido. Las medianas, medias y desviaciones se calculan solo con
entrenamiento. Hacerlo con toda la muestra introduciría información de la
evaluación en la preparación.

## 11. Qué es una red neuronal en este proyecto

Es una función formada por capas que combinan números. Una neurona de una
capa Linear calcula una suma de entradas multiplicadas por pesos y añade
un sesgo. Los pesos y sesgos son parámetros que se aprenden.

Una fórmula pequeña, solo para entender una neurona, sería:

`resultado = entrada1 × peso1 + entrada2 × peso2 + sesgo`

Nuestra configuración elegida hace este recorrido: **5 → 64 → 32 → 4**.
Cinco entradas, dos capas ocultas y cuatro salidas. Una capa oculta realiza
transformaciones intermedias; no son nuevas columnas conocidas del CSV.

```python
modelo = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 4)
)
```

`Sequential` aplica las operaciones por orden. ReLU convierte los resultados
negativos en cero y mantiene los positivos. Introduce no linealidad: permite
aprender relaciones que no son una simple combinación lineal de las entradas.

La red tiene 2.596 parámetros: (5×64 + 64) + (64×32 + 32) + (32×4 + 4).
Los números añadidos a cada producto son los sesgos.

## 12. Tensores y predicciones

Un tensor es una estructura numérica parecida a un array. PyTorch realiza
operaciones con tensores y calcula gradientes para entrenar.
Nuestras entradas usan float32 y las etiquetas, enteros int64.

Con un lote de 32 películas:

| Momento | Forma |
| --- | --- |
| Entrada | 32 × 5 |
| Primera capa oculta | 32 × 64 |
| Segunda capa oculta | 32 × 32 |
| Salida | 32 × 4 |

Cada fila sigue representando una película. El último lote puede ser menor.

Las salidas son puntuaciones llamadas logits, no porcentajes. Un ejemplo
inventado sería `[0.2, -0.5, 0.8, 1.4]`. La puntuación mayor está en la
posición 3, que corresponde a Alta. `argmax(1)` escoge esa posición para
cada película del lote.

## 13. Cómo aprende

Al principio los parámetros se inicializan y las predicciones todavía no
están ajustadas al problema. Se calculan errores usando las etiquetas reales
de entrenamiento y se actualizan los parámetros para reducirlos.

```python
optimizador.zero_grad()
salida = modelo(x_lote)
error = perdida(salida, y_lote)
error.backward()
optimizador.step()
```

1. `zero_grad()` limpia los gradientes anteriores, que PyTorch acumula.
2. El modelo calcula cuatro puntuaciones por película.
3. `CrossEntropyLoss` mide el error respecto a las etiquetas correctas.
4. `backward()` calcula gradientes: cómo influye cada parámetro en el error.
5. Adam utiliza esos gradientes para actualizar los parámetros.

`backward()` calcula; `step()` actualiza. La tasa de aprendizaje controla
el tamaño de los ajustes junto con el funcionamiento de Adam. No es un
porcentaje de aciertos.

No añadimos Softmax antes de CrossEntropyLoss: esta pérdida recibe logits
y ya incorpora internamente el cálculo necesario.

Una época recorre todas las películas de entrenamiento. Un batch o lote es
el grupo usado en una actualización. Con 893 películas y batch 32 hay 28
lotes por época, con el último incompleto. Los experimentos hacen 15 épocas.

Durante evaluación se utiliza `eval()` y `torch.no_grad()`. No actualizamos
pesos con validación ni con prueba. En estas redes sin Dropout o BatchNorm,
`eval()` no cambia las capas, pero explicita el modo de uso.

## 14. Qué hemos probado

Un hiperparámetro es una decisión que fijamos, como el número de neuronas,
el lote o la tasa de aprendizaje. Los pesos son parámetros que aprende la red.

| Configuración | Capas ocultas | Batch | Learning rate |
| --- | --- | --- | --- |
| Exp_1 | 32 | 32 | 0,01 |
| Exp_2 | 64–32 | 32 | 0,01 |
| Exp_3 | 128–64–32 | 32 | 0,01 |
| Exp_4 | 64–32 | 16 | 0,01 |
| Exp_5 | 64–32 | 64 | 0,01 |
| Exp_6 | 64–32 | 32 | 0,001 |
| Exp_7 | 64–32 | 32 | 0,0001 |
| Exp_8 | 64 | 32 | 0,01 |

Exp_1 frente a Exp_8 estudia anchura. Exp_8 frente a Exp_2 añade una capa.
Exp_2 frente a Exp_3 cambia capas y tamaños a la vez: no permite atribuir
la diferencia exclusivamente a uno de los dos. Exp_2/4/5 cambia batch;
Exp_2/6/7 cambia learning rate.

Cada configuración se repite con semillas 42, 7 y 21. Cambian la inicialización
y el barajado, manteniendo las películas de cada conjunto. Son 24 entrenamientos;
no es validación cruzada. Aparte se conserva el modelo inicial del equipo:
dos capas de 64, diez épocas y evaluación solo en validación.

Tenemos dos decisiones distintas:

| Decisión | Criterio |
| --- | --- |
| Qué época de cada entrenamiento guardar | Menor pérdida de validación. |
| Qué configuración escoger | Mayor F1 macro medio de validación de las tres semillas. |

Guardamos una copia de los mejores pesos con `copy.deepcopy(state_dict())`
para que las siguientes épocas no alteren ese estado guardado.

## 15. Cómo medimos el resultado

Accuracy es aciertos dividido entre ejemplos. Para la semilla 42 son
117/192 = 0,609375, que redondea a **60,9 %**.

F1 combina precisión y recall de una clase:

- Precisión: de lo que el modelo etiqueta como esa clase, cuánto era correcto.
- Recall: de los casos que realmente pertenecían a esa clase, cuántos detectó.
- F1: 2 × precisión × recall / (precisión + recall).
- F1 macro: media del F1 de las cuatro clases, con el mismo peso para cada una.

Por ejemplo, para Alta en la matriz hay 42 aciertos, 52 predicciones Alta y
49 películas realmente Alta. Su precisión es 42/52 ≈ 0,808 y su recall,
42/49 ≈ 0,857. Su F1 es aproximadamente 0,832. Después se promedia el F1 de
Alta con los de Baja, Media y Media-alta para calcular el macro.

La pérdida de entrenamiento no es accuracy ni F1. Utiliza las puntuaciones
para aprender, no solo cuenta las clases acertadas.

## 16. Qué resultados hemos obtenido

La elegida es **Exp_2**, por F1 macro medio de validación ≈ 0,633. Su accuracy
media de validación es ≈ 63,7 %. No confundas estas dos métricas.

En prueba, el promedio de las tres semillas da **60,9 % de accuracy** y
**F1 macro 0,603**, con desviación de F1 ≈ 0,002. Es la variación de tres
inicializaciones sobre una partición; no un intervalo de confianza.
La matriz y los ejemplos utilizan siempre la semilla 42, fijada de antemano.

| Modelo o referencia | Accuracy de prueba |
| --- | --- |
| Clase mayoritaria de entrenamiento | 25 % |
| Regresión logística | 58,9 % |
| Red elegida, promedio de tres semillas | 60,9 % |

La regresión logística, pese a su nombre, es un clasificador. Usa las mismas
cinco entradas y los mismos conjuntos. La mejora de la red es pequeña:
aproximadamente 2,1 puntos porcentuales de accuracy y 0,015 de F1 macro.
No hemos demostrado una superioridad general ni significación estadística.

La clase mayoritaria se decide con entrenamiento. Hay empate entre Media-alta
y Alta con 225 ejemplos; el código escoge la primera en el orden fijado,
Media-alta. En prueba tiene 48 casos, por lo que acierta 48/192 = 25 %.

La red más grande, Exp_3, da F1 de validación ≈ 0,594. Más capas no aseguran
mejorar. Exp_7 da ≈ 0,373 con tasa 0,0001 y quince épocas; puede estar
aprendiendo despacio. No demuestra que esa tasa nunca funcione.

El modelo inicial de 64–64 obtiene 60,7 % de accuracy en validación. Esa
cifra corresponde a otro modelo y otro conjunto de evaluación: no la
compares directamente con 60,9 % en prueba.

## 17. Leer la matriz y los ejemplos

Las filas son categorías reales y las columnas son predicciones.

| Real / predicha | Baja | Media | Media-alta | Alta |
| --- | --- | --- | --- | --- |
| Baja | 32 | 13 | 2 | 0 |
| Media | 15 | 21 | 11 | 1 |
| Media-alta | 5 | 12 | 22 | 9 |
| Alta | 0 | 0 | 7 | 42 |

La diagonal contiene 32 + 21 + 22 + 42 = 117 aciertos. El mayor error fuera
de la diagonal es Media → Baja, con quince películas. La matriz describe
qué errores ocurrieron; por sí sola no demuestra sus causas.

Ejemplos reales del CSV de prueba:

| Película | Real | Predicción | Resultado |
| --- | --- | --- | --- |
| Shrek 2 | Alta | Alta | Acierto. |
| The Incredibles | Alta | Alta | Acierto. |
| Gangster Squad | Media-alta | Alta | Error. |
| Boys Don't Cry | Media-alta | Baja | Error. |

Son ejemplos ilustrativos, no una muestra representativa. No se utilizan
para escoger la configuración. El modelo no recibe sus títulos.

## 18. Curvas y sobreajuste

Sobreajustar significa aprender demasiado las particularidades del conjunto
de entrenamiento y generalizar peor a otros ejemplos. Una señal sería que
la pérdida de entrenamiento baje y la de validación empeore de forma sostenida.
No basta una oscilación aislada para afirmarlo.

En las curvas mira el eje horizontal, las épocas; el vertical, la pérdida
o la accuracy; y qué línea corresponde a cada conjunto. Las curvas muestran
todas las épocas, aunque recuperemos pesos de una época anterior.

La figura de comparación usa F1 medio de validación y barras de desviación
entre semillas. La ventaja de Exp_2 sobre Exp_5 es ≈ 0,0087 de F1, frente a
desviaciones ≈ 0,0099 y 0,0140. Se elige por el criterio declarado, reconociendo
que la diferencia es pequeña.

## 19. Para qué sirve cada archivo

| Archivo | Qué debes saber explicar |
| --- | --- |
| EDA.py | Lee, limpia, guarda el CSV limpio y genera estadísticas y cinco gráficos del EDA. |
| preparacion_datos_base.py | Codifica etiquetas, separa conjuntos, imputa si hace falta y escala sin mezclar datos. |
| preparacion_datos_entrenamiento.py | Entrena el modelo inicial 64–64 durante diez épocas. |
| preparacion_hiperparametros_modulares.py | Construye las redes, hace los 24 entrenamientos, selecciona y evalúa. |
| generar_informe.py | Convierte métricas guardadas en tablas, ejemplos, conclusiones y figuras. |
| comprobar_fuente.py | Auditoría opcional frente al ZIP público de Kaggle. |
| presentacion.qmd | Texto y código explicativo del informe Quarto. |
| resultados/ | CSV de métricas, particiones, predicciones, historiales y pesos .pt. |
| graficos/ | Figuras PNG para el informe. |

La clase modular hereda de `nn.Module`. Su constructor añade una `Linear`
y una `ReLU` por cada tamaño de la lista, y termina con cuatro salidas.
`nn.Sequential(*layers)` pasa los elementos de la lista como argumentos.
`forward` define cómo atraviesan la red las entradas.

Los pesos `.pt` permiten volver a cargar una red entrenada. Para usarla con
datos nuevos también se necesitan la misma arquitectura, orden de variables,
preprocesado y orden de etiquetas. Un archivo de pesos solo no lo describe todo.

## 20. Qué hace Quarto y cómo se reproduce

Quarto convierte el documento `.qmd`, las tablas y las figuras en una web.
PyTorch entrena; el navegador muestra el informe. Los bloques del informe
explican el código: la ejecución completa se hace con los scripts.

Con el entorno activado, desde `redes neuronales`:

```bash
python EDA.py
python preparacion_hiperparametros_modulares.py
quarto render
```

Primero se limpian los datos; después se entrena y generan resultados;
por último Quarto incorpora esas salidas al informe. Para exponer los
resultados ya calculados basta abrir `presentacion.html`, la copia actualizada
con recursos incrustados. El entrenamiento no se ejecuta al abrir ese HTML.

El código está organizado dentro de `redes neuronales`. La configuración de
Actions está en `.github/workflows/` porque GitHub necesita esa ubicación.
La publicación requiere completar la configuración de Pages y del entorno;
después hay que entregar la URL pública del informe en Loop.

## 21. Limitaciones que hay que reconocer

- Muestra pequeña, histórica y previamente seleccionada por casos completos.
- Información posterior al estreno, relacionada con el propio indicador.
- Un único reparto de películas: tres semillas no prueban generalización entre muestras.
- Ventaja pequeña frente a la regresión logística.
- Etiquetas propias de la muestra; no categorías oficiales del catálogo.
- Presupuesto y recaudación superiores a 10.000 excluyen películas válidas.
- Tratar las clases como nominales cuenta igual confundir una categoría vecina
  que confundir extremos. No hemos usado una pérdida ordinal.

Mejoras futuras razonables: datos más recientes, validación cruzada o temporal,
comparar sin número de votos y estudiar más épocas usando validación.
No se deben ajustar decisiones repetidamente mirando la prueba final.

## 22. Respuestas cortas para la defensa

**¿Qué hace vuestro proyecto?** Clasifica películas en cuatro niveles de
popularidad con cinco características y compara redes de PyTorch.

**¿Por qué es clasificación?** La salida es una categoría; no una cifra de recaudación.

**¿Por qué una red?** Permite probar relaciones no lineales entre cinco
variables. La comparamos con una referencia sencilla porque su superioridad
no está garantizada.

**¿Por qué excluís popularity?** Porque determina directamente la categoría objetivo.

**¿Por qué tres conjuntos?** Para aprender, elegir configuración y evaluar
al final sin utilizar las mismas películas para las tres tareas.

**¿Por qué normalizáis?** Para facilitar el aprendizaje con columnas de
escalas diferentes. Los parámetros se calculan solo con entrenamiento.

**¿Qué aprende la red?** Sus pesos y sesgos. Nosotros fijamos los hiperparámetros.

**¿Qué hace backward?** Calcula gradientes. El optimizador actualiza los pesos.

**¿Por qué elegís Exp_2?** Tiene el mayor F1 macro medio de validación,
según el criterio declarado.

**¿Es bueno 60,9 %?** Es un resultado moderado: mejora poco a la logística
y supera la referencia mayoritaria. El enunciado no exige una accuracy concreta.

**¿Más capas mejoran?** No necesariamente: la red más grande no mejoró aquí.

**¿Qué falta para entregar?** Publicar y comprobar la web, poner la URL en
Loop, verificar el registro del equipo y exponer los cuatro.

## 23. Repaso final sin leer

Explica este recorrido en tus palabras:

CSV → exploración y limpieza → tres conjuntos → preprocesado aprendido
con entrenamiento → redes PyTorch → elección con validación → evaluación
final en prueba → informe Quarto.

Después abre una tabla, una curva y la matriz. Para cada una di qué representan
los ejes o columnas, qué resultado observas y qué limitación tiene.
Si puedes hacer eso y explicar las cinco líneas del entrenamiento, conoces
el núcleo del proyecto. Utiliza `GUIA_EXPOSICION.md` para ensayar el reparto
entre los cuatro.
