# Trabajo 1 · Cine y redes neuronales

Equipo: José Ramón, Guillermo, Dylan y Tomás. 2.º DAW.

## Para preparar la exposición de mañana

1. Abrir `presentacion.html`: informe completo con recursos incrustados, para leerlo sin ejecutar Python.
2. Repartir `GUIA_EXPOSICION.md`: incluye resultados concretos, ejemplos y guion para los cuatro.
3. Dylan puede estudiar `EXPLICACION_DYLAN.md` para entender la red y el bucle paso a paso.
4. Revisar `ANTES_DE_PRESENTAR.md`: ensayo, apertura local y publicación pendiente.
5. Leer `EXPLICACION_COMPLETA.md` para entender el recorrido desde cero.
6. Abrir `repaso_proyecto.html` con Chrome, Edge o Safari y pulsar **Reproducir**.
   Incluye 25 capítulos, explicaciones visuales y el guion completo. La opción
   **Leer con voz** depende de las voces disponibles en el navegador; el texto
   y las imágenes funcionan sin conexión. Es un repaso interactivo en HTML.

## Estado de esta versión

La base de esta revisión es la rama **`desarrollo`**, commit
`0857c4b05a69e11ffdf3df62ee57eaf9d91ebad2`. La copia recibida inicialmente
era de `main`, que tenía menos archivos. La versión original de desarrollo
se conserva íntegra en `originales/desarrollo/`.

Se mantienen las cinco variables y las siete configuraciones del equipo.
Se añade una octava comparación de anchura, tres semillas y una prueba final.
La muestra contiene 1.276 películas después del umbral comercial heredado
y de eliminar una categoría ausente: 893 de entrenamiento, 191 de validación
y 192 de prueba. Se han ejecutado el modelo inicial y los 24 entrenamientos,
y se ha generado el HTML con Quarto. Los pesos guardados reproducen las
predicciones y no hay películas compartidas entre los conjuntos.

El entrenamiento y los gráficos usan las dependencias de `requirements.txt`.
El informe indica explícitamente si no hay métricas de una ejecución completa.
Se ha contrastado una fuente pública en Kaggle, reproducido la selección de
1.287 identificadores y las etiquetas, y documentado las diferencias en
`datos/fuente.md`. El EDA corrige 12 fechas con el siglo incorrecto; las cinco
entradas de la red no cambian. Se conserva el CSV recibido sin modificar.

## Instalación en macOS

Abrir Terminal en la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "redes neuronales/requirements.txt"
```

Si no está instalado Quarto:

```bash
brew install --cask quarto
```

También está el instalador oficial: https://quarto.org/docs/get-started/

Comprobar la instalación:

```bash
python -c "import torch, pandas, numpy, matplotlib, sklearn; print('PyTorch:', torch.__version__)"
quarto --version
```

Si no se encuentra una distribución compatible con la versión de Python,
usar Python 3.12 en un entorno nuevo. No es necesario reinstalar el sistema
ni modificar el Python global. GitHub Actions usa Python 3.12 y Colab es otra
alternativa si la instalación local falla.

## Ejecutar el proyecto

Con el entorno virtual activado, desde la raíz:

```bash
cd "redes neuronales"
python EDA.py
python preparacion_hiperparametros_modulares.py
quarto render
quarto preview
```

`preparacion_hiperparametros_modulares.py` ejecuta el modelo inicial y 24
entrenamientos: ocho configuraciones, tres semillas cada una. Usa CPU;
no necesita una tarjeta gráfica. La duración
depende del equipo. Imprime el resultado de cada entrenamiento al terminarlo.

El HTML queda en `_site/index.html`. Abrirlo con Quarto o con el navegador.
El HTML incluye las figuras y recursos para poder guardarlo como copia local
de la exposición. No sustituye la URL pública que exige la entrega.

Si se interrumpe el entrenamiento, volver a ejecutarlo completo. No mezclar
CSV de distintas ejecuciones. Tras cambiar los datos o el código, repetir
EDA, entrenamiento y render en ese orden.

## Archivos principales

| Archivo o carpeta | Función |
| --- | --- |
| `EDA.py` | Limpieza, tablas descriptivas y cinco gráficos del EDA. |
| `preparacion_datos_base.py` | Cinco entradas, separación y normalización sin mezclar conjuntos. |
| `preparacion_datos_entrenamiento.py` | Modelo inicial: dos capas de 64 y diez épocas. |
| `preparacion_hiperparametros_modulares.py` | Ocho configuraciones, tres semillas y evaluación final. |
| `generar_informe.py` | Figuras y texto de resultados a partir de las métricas reales. |
| `comprobar_fuente.py` | Comparación opcional con el ZIP público de Kaggle. |
| `presentacion.qmd` y `_quarto.yml` | Fuente y configuración del informe Quarto. |
| `datos/` | CSV recibido, CSV limpio y notas de procedencia. |
| `originales/` | Script y CSV limpio de la versión inicial, sin ejecutar. |
| `resultados/` | Tablas, divisiones, predicciones, curvas numéricas y pesos entrenados. |
| `graficos/` | Imágenes generadas con matplotlib. |
| `GUIA_EXPOSICION.md` | Reparto, explicación del código y preguntas de defensa. |
| `EXPLICACION_DYLAN.md` | Explicación pausada de la red y el entrenamiento. |
| `EXPLICACION_COMPLETA.md` | Guía de todo el proyecto desde cero, con resultados y preguntas. |
| `repaso_proyecto.html` | Repaso visual con reproducción, pausa y selección de capítulos. |
| `ANTES_DE_PRESENTAR.md` | Preparación local, ensayo y pasos de entrega pendientes. |
| `Ejecutar_en_Colab.ipynb` | Alternativa de ejecución, después de subir esta versión al repo. |

`--sin-graficos` permite ejecutar la limpieza sin matplotlib, pero no genera
las figuras requeridas para la entrega. No es el modo de ejecución final.

## Publicar con GitHub Pages

La raíz incluye `.github/workflows/informe.yml`. Es la excepción de ubicación
necesaria para que GitHub encuentre la configuración de publicación; el
código, datos y fuente del informe están dentro de `redes neuronales`.

1. Trabajar sobre la clonación de `desarrollo` y subir los cambios a esa rama.
   La primera revisión se subió en `2f5399d`. Incluir `.github/` y conservar
   las copias de `originales/desarrollo/`. No subir `.venv/` ni `_site/`.
2. En el repositorio, abrir **Settings → Pages** y elegir **GitHub Actions**
   como fuente de publicación.
3. En **Actions**, abrir **Informe de redes neuronales** y ejecutar **Run workflow**
   en `desarrollo` si la subida ya se hizo antes de activar Pages.
   Esta entrega se sube a la rama `desarrollo`, en minúsculas.
4. El proceso instala dependencias, genera EDA, entrena, renderiza Quarto y publica.
   Si el entorno `github-pages` tiene restricciones de ramas, debe permitir
   `desarrollo` para completar el despliegue.
5. Cuando termine en verde, abrir la URL mostrada por el despliegue y comprobar
   que aparecen los gráficos, la tabla de ocho configuraciones y la matriz de confusión.
6. Publicar ESA URL en Loop. El enlace al repositorio es un enlace diferente.

El workflow también guarda un ZIP de resultados como artefacto descargable
desde la ejecución de Actions. No hace commits automáticos ni modifica el repo.
Un fallo se debe revisar en el paso que aparezca en rojo; no dar la publicación
por hecha solo por haber subido los archivos.

## Antes de entregar

- Consultar la fuente contrastada y `datos/fuente_verificada.json`.
- Verificar que el entrenamiento completo termina sin errores.
- Leer las conclusiones que se generen y contrastarlas con las tablas.
- Revisar la web publicada desde otro dispositivo.
- Comprobar el registro de los cuatro integrantes y del repositorio en Loop.
- Publicar la URL del informe antes del límite y ensayar las cuatro intervenciones.
