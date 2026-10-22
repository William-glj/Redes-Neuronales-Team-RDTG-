Esta revisión vuelve a partir del commit 0857c4b, el que teníamos antes de los cambios de hoy. Se recuperan los cuatro scripts, sus clases y funciones, las siete configuraciones y la estructura y explicaciones de Fase A, B y C en presentacion.qmd. Los archivos se colocan dentro de redes neuronales, como pide el trabajo, manteniendo los nombres originales.

Se han añadido los gráficos que faltaban al EDA, la revisión de nulos y atípicos, y la fuente pública contrastada. Se corrigen doce fechas y se excluye una película sin categoría para evitar una quinta clase nan. Quedan 1.276 películas. Las cinco entradas siguen siendo presupuesto, duración, votos, nota media y año; se mantiene StandardScaler.

Para evaluar, reservamos 893 películas para entrenamiento, 191 para validación y 192 para prueba. Las siete configuraciones se repiten con tres semillas: 21 entrenamientos. Elegimos por accuracy media de validación al terminar quince épocas y después evaluamos en prueba.

En esta ejecución gana Exp_1, una capa oculta de 32, lote 32 y tasa 0,01. Su accuracy media es 61,78 % en validación y 67,53 % en prueba; F1 macro de prueba 0,673. Las cifras anteriores pertenecen a otros experimentos y no deben mezclarse. La regresión logística obtiene 63,54 % en la misma prueba.

Las tablas y gráficos se generan desde los CSV guardados. Para verlo, abrid redes neuronales/presentacion.html. Para comprobar el código, siguen los cuatro archivos con los mismos nombres; CAMBIOS.md explica las modificaciones concretas desde 0857c4b. GUIA_EXPOSICION.md reparte las cuatro partes. Falta comprobar la publicación de GitHub Pages y entregar su URL en Loop.
