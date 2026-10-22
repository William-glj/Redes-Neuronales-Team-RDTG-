# Datos utilizados

- Archivo recibido: `TMBD Movie Dataset.csv`.
- Nombre actual: `peliculas_original.csv`; se ha renombrado sin modificar sus bytes.
- Copia aportada en el repositorio del equipo:
  https://github.com/William-glj/Redes-Neuronales-Team-RDTG-/tree/desarrollo
- Revisión de referencia: `0857c4b05a69e11ffdf3df62ee57eaf9d91ebad2`.
- Dimensiones recibidas: 1.287 filas y 24 columnas, incluyendo un índice exportado.
- Periodo observado: 1961–2015.

## Pendiente de confirmar por quien descargó los datos

1. URL exacta de descarga original, autor y licencia o condiciones.
2. Cómo se seleccionaron estas 1.287 películas.
3. Cómo se construyó `popularity_level`.

La estructura coincide con una selección de datos históricos de TMDB, pero eso
no acredita el origen exacto. No atribuir este CSV al dataset TMDB 5000 de Kaggle:
no hemos comprobado esa equivalencia.

Se conservan las etiquetas recibidas. Sus rangos son compatibles con cuartiles,
pero no se presenta esa deducción como un procedimiento original comprobado.

Referencia oficial para entender el indicador (no es el enlace de descarga):
https://developer.themoviedb.org/docs/popularity-and-trending
