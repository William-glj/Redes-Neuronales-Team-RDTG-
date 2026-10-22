# Trabajo 1: Cine y redes neuronales

Equipo: José Ramón, Guillermo, Dylan y Tomás. 2.º DAW.

Esta revisión parte de `0857c4b`. Conserva los cuatro scripts, sus clases y funciones,
las siete configuraciones y los apartados 1, 2 y 3 del informe del grupo. Los archivos
se colocan dentro de `redes neuronales`, como pide el enunciado.

## Abrir el informe

Abrir `presentacion.html` con Chrome, Edge o Safari. Incluye las figuras y puede leerse
sin ejecutar Python. La fuente de Quarto sigue siendo `presentacion.qmd`.

## Ejecutar

Desde la raíz del repositorio, activar un entorno con las dependencias de `requirements.txt`.
Después, dentro de `redes neuronales`:

```bash
python EDA.py
python preparacion_datos_entrenamiento.py
python preparacion_hiperparametros_modulares.py
python generar_informe.py
quarto render
```

El último comando genera `_site/index.html`. Para actualizar la copia de exposición,
volver a copiar ese HTML a `presentacion.html`.

Las versiones usadas están en `requirements.txt`. No hace falta GPU. Las tablas se generan
con Python y se incorporan a Quarto; no se necesita Jupyter para renderizar este informe.

## Archivos del grupo

- `EDA.py`: limpieza, diagnóstico y cinco gráficos del EDA.
- `preparacion_datos_base.py`: cinco entradas, LabelEncoder, separación y StandardScaler.
- `preparacion_datos_entrenamiento.py`: modelo inicial 64–64, diez épocas.
- `preparacion_hiperparametros_modulares.py`: las siete configuraciones y sus tres semillas.
- `presentacion.qmd`: informe con la estructura y las explicaciones del grupo.
- `TMBD Movie Dataset.csv`: archivo recibido, conservado sin modificar.
- `IMBD_Movie_Dataset_limpio.csv`: salida del EDA.
- `generar_informe.py`: tablas y figuras a partir de los resultados guardados.
- `resultados/` y `graficos/`: evidencias de las ejecuciones.
- `datos/fuente.md`: procedencia y comprobación de la fuente pública.

## Resultado de esta revisión

Se elige Exp_1 dentro de la rejilla: una capa oculta de 32 neuronas, lote 32 y tasa 0,01.
Su accuracy media es 61,78 % en validación y 67,53 % en prueba; F1 macro de prueba 0,673.
La regresión logística obtiene 63,54 % en la misma prueba. Se reserva prueba antes de entrenar
para no utilizarla al elegir la configuración. Las cifras antiguas corresponden a otro protocolo.

Consultar `CAMBIOS.md` para las modificaciones concretas desde `0857c4b` y
`GUIA_EXPOSICION.md` para repartir la defensa.

## Publicación

Los cambios se suben a `desarrollo`. El workflow de `.github/workflows/informe.yml`
ejecuta los scripts y publica el informe con GitHub Pages. Un administrador debe elegir
GitHub Actions como fuente de Pages y permitir `desarrollo` en el entorno `github-pages`.
Una subida al repositorio no demuestra que el despliegue haya terminado.
La entrega exige comprobar la web y publicar su URL en Loop.
