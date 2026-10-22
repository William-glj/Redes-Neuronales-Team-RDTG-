# Datos utilizados y comprobación de procedencia

## Fuente pública contrastada

- Dataset: **TMDb Movies Dataset**.
- Publicador: **Juzer Shakir**, usuario `juzershakir` de Kaggle.
- Ficha: https://www.kaggle.com/datasets/juzershakir/tmdb-movies-dataset
- Descarga pública: https://www.kaggle.com/api/v1/datasets/download/juzershakir/tmdb-movies-dataset
- Versión contrastada: 1; archivo `tmdb_movies_data.csv`, 10.866 filas y 21 columnas.
- Licencia que indica la ficha consultada: **Unknown**. No le atribuimos una
  licencia abierta que no aparece documentada.

El archivo recibido por el equipo se llamaba `TMBD Movie Dataset.csv` y ya
estaba en la rama `desarrollo`, revisión `0857c4b`. Se conserva sin cambiar
sus bytes como `TMBD Movie Dataset.csv` dentro de `redes neuronales`.
Tiene 1.287 películas y 24 columnas contando el índice exportado.

## Qué se ha comprobado

Los mismos **1.287 identificadores** se obtienen del CSV público al convertir
en nulos los ceros de presupuesto y recaudación, eliminar filas con nulos y
quitar duplicados por `id`. Esto reproduce la selección recibida; no acredita
el enlace que utilizó entonces quien descargó la copia ni conserva su código.

Se han contrastado las 21 columnas de la fuente, con estas diferencias documentadas:

- Las fechas tienen otro formato y 12 años tienen el siglo incorrecto en la
  copia recibida. Mes y día coinciden. `release_year` sí coincide con la fuente.
  El EDA corrige esas fechas solo en el CSV limpio.
- `budget_adj` y `revenue_adj` tienen distinto número de decimales; coinciden
  con tolerancia relativa de 1e-9. No se utilizan como entradas de la red.
- En el resto de números se permite únicamente el redondeo de lectura de CSV
  (tolerancia relativa y absoluta de 1e-12).
- `profit` es una columna añadida y coincide con `revenue - budget`.
- `popularity_level` es otra columna añadida. Las cuatro etiquetas y el nulo
  se reproducen con `pd.cut` usando mínimo, cuartiles y máximo de la
  popularidad de las 1.287 películas. El mínimo queda fuera del primer
  intervalo si no se usa `include_lowest=True`. Conservamos las etiquetas
  recibidas y eliminamos la fila sin etiqueta.

Los cortes reproducidos son: 0,010335; 0,6647835; 1,152354; 2,1253415; 32,985763.
Son cortes de esta selección histórica, no umbrales oficiales de TMDB.

La selección previa de casos completos reduce mucho la muestra y puede
favorecer películas bien documentadas. No representa el catálogo completo.

## Cómo repetir la comprobación

Descargar el ZIP de Kaggle y, desde `redes neuronales`, ejecutar:

```bash
python comprobar_fuente.py /ruta/al/archivo.zip
```

No necesita credenciales de TMDB y no cambia el CSV recibido. El resultado,
los hashes SHA-256 y los cortes quedan en `datos/fuente_verificada.json`.
El script no forma parte del entrenamiento: sirve para auditar la procedencia.

Referencia para entender el indicador, diferente del descargable histórico:
https://developer.themoviedb.org/docs/popularity-and-trending
