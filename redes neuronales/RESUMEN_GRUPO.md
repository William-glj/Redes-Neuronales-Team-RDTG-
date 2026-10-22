Chicos, os resumo los cambios para que sea más fácil ubicarse al abrir el proyecto.

La base es lo que ya teníamos en `desarrollo`: se mantienen el tema de cine, las cinco variables, el modelo inicial y las siete configuraciones que habíais preparado. La entrega se ha organizado dentro de `redes neuronales`, como pide el enunciado, y hay una copia de la versión anterior en `originales/desarrollo`.

Se ha completado el EDA con estadísticas y gráficos y se ha corregido una película sin categoría que podía acabar contando como una quinta clase llamada "nan". Tras el filtro comercial que ya usábamos, quedan 1.276 películas.

Para entrenar seguimos usando presupuesto, duración, número de votos, nota media y año. Se han separado entrenamiento, validación y prueba final, para elegir la red sin utilizar las películas con las que después medimos el resultado final. Se añade una octava configuración para comparar el número de neuronas y cada configuración se repite con tres semillas: 24 entrenamientos, además del modelo inicial.

La configuración elegida tiene dos capas ocultas de 64 y 32 neuronas, lote 32 y tasa de aprendizaje 0,01. En la prueba final consigue un 60,9 % de aciertos. La referencia sencilla de regresión logística consigue un 58,9 %. No hay que comparar directamente ese porcentaje con el que aparecía antes: el anterior era de validación y ahora hemos cambiado el reparto y la preparación.

Para orientarse con los archivos:

- `EDA.py`: limpieza, estadísticas y gráficos.
- `preparacion_datos_base.py`: preparación y separación de datos.
- `preparacion_datos_entrenamiento.py`: modelo inicial de dos capas de 64.
- `preparacion_hiperparametros_modulares.py`: experimentos y evaluación final.
- `presentacion.qmd`: informe Quarto con explicación, código y resultados.
- `GUIA_EXPOSICION.md`: reparto sugerido y preguntas para preparar la defensa.

Las tablas y gráficas se generan a partir de las ejecuciones para que coincidan con el código. Las métricas y predicciones están en `resultados`, así que no hace falta entrenarlo en directo durante la exposición.

La primera revisión ya se subió a `desarrollo` en el commit `2f5399d`. GitHub ejecutó bien el análisis, los entrenamientos y Quarto; la publicación de la web se bloqueó porque el entorno solo permite la rama `gh-pages`. Un administrador debe configurar Pages con GitHub Actions y permitir `desarrollo`.

En la revisión local final se ha localizado una fuente pública en Kaggle y se han contrastado los 1.287 identificadores, las columnas y las etiquetas. Esa comprobación detectó 12 fechas con el siglo incorrecto, que se corrigen en el CSV limpio; el año que usa la red ya era correcto. Los detalles están en `datos/fuente.md`.

La guía de exposición incluye ya las cifras concretas, el reparto y un guion para los cuatro. `EXPLICACION_DYLAN.md` explica la red paso a paso y `ANTES_DE_PRESENTAR.md` recoge la preparación para mañana. El informe añade ejemplos reales de aciertos y errores. `EXPLICACION_COMPLETA.md` recorre todo desde cero y `repaso_proyecto.html` permite estudiar con 25 capítulos visuales: se abre en el navegador y se pulsa Reproducir. Empezad por `GUIA_EXPOSICION.md` y abrid `presentacion.html` para ensayar. Esta revisión se sube a `desarrollo`; queda revisar entre todos, comprobar la web y entregar su URL en Loop.
