# Revisión desde 0857c4b

## Base conservada

Se han extraído directamente los archivos de `0857c4b`: tema de cine, CSV recibido,
cuatro scripts, clases, funciones, cinco entradas, modelo inicial 64–64 y siete
configuraciones. El informe vuelve a partir de la redacción y los apartados del grupo:
Fase A, Fase B y Fase C. Se mantienen sus explicaciones de neuronas, DataLoader,
evaluación y comparación de hiperparámetros.

## Correcciones y ampliaciones

1. Código y recursos dentro de `redes neuronales`, manteniendo los nombres originales.
2. Se excluye una etiqueta ausente antes de LabelEncoder para evitar una clase `nan`.
3. Se contrastan la fuente pública y doce fechas con el siglo incorrecto. El año de entrada ya era correcto.
4. Se añaden tablas de nulos, IQR y cinco gráficos al EDA. Los extremos válidos se conservan.
5. Se reserva una prueba final: aproximadamente 70/15/15, con estratificación y semilla 42.
6. Se mantiene StandardScaler ajustado solo con entrenamiento. No se añaden logaritmos a las entradas.
7. Se repiten las siete configuraciones originales con semillas 42, 7 y 21: 21 entrenamientos.
8. Se elige por accuracy media de validación al terminar las quince épocas; desempate por menor pérdida media.
9. Se guardan pesos, curvas, métricas y predicciones. La prueba se usa después de elegir.
10. Las salidas escritas a mano del informe se sustituyen por tablas y figuras de la ejecución.
11. Se corrigen la justificación de la red, el orden real de LabelEncoder y conclusiones causales no demostradas.
12. Se añaden limitaciones, prueba independiente, referencias sencillas y pasos para reproducir.

## Cifras actuales

- 1.287 películas recibidas; 1.276 después de limpiar.
- 893 entrenamiento, 191 validación, 192 prueba.
- Modelo inicial: 62,30 % de accuracy de validación, una semilla.
- Exp_1 seleccionada entre las siete: 61,78 % de accuracy media de validación.
- Prueba de Exp_1: 67,53 % de accuracy media y F1 macro 0,673.

La evaluación y los resultados son propios de esta revisión. No deben mezclarse
con los del informe antiguo ni con revisiones que utilizaran otra preparación.
