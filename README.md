# Cine y redes neuronales · Equipo RDTG

Proyecto de Desarrollo de aplicaciones web en entorno cliente, 2.º DAW.

**Integrantes:** José Ramón, Guillermo, Dylan y Tomás.

La primera entrega está dentro de [`redes neuronales`](redes%20neuronales/README.md).
Estudiamos películas y comparamos pequeñas redes de PyTorch para clasificar su
popularidad en cuatro categorías. Es una clasificación con datos posteriores
al estreno, no una predicción de éxito antes del estreno.

Consulta allí las instrucciones de ejecución, publicación y exposición.

## Ejecutar desde esta carpeta

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "redes neuronales/requirements.txt"
cd "redes neuronales"
python EDA.py
python preparacion_hiperparametros_modulares.py
quarto render
quarto preview
```

Quarto debe estar instalado (`brew install --cask quarto` o su instalador
oficial). Las instrucciones detalladas están en `redes neuronales/README.md`
y la preparación de la defensa en `redes neuronales/GUIA_EXPOSICION.md`.

**Comprobado:** limpieza, modelo inicial, 24 entrenamientos, separación de
datos, predicciones de los pesos guardados y generación del HTML con Quarto.
La fuente pública se ha contrastado con Kaggle y se documentan sus diferencias.
Para preparar la defensa, empezar por `redes neuronales/GUIA_EXPOSICION.md`
y abrir la copia local `redes neuronales/presentacion.html`.

**Pendiente antes de entregar:** revisión por el equipo y completar la
publicación del informe. El workflow de `.github/workflows/informe.yml` prepara y publica
la web cuando se suben los cambios a `desarrollo` y se activa GitHub Pages con Actions.
El entorno `github-pages` también debe permitir esa rama; la primera publicación
quedó bloqueada por esa regla. Ver `redes neuronales/ANTES_DE_PRESENTAR.md`.

Esta revisión continúa la rama `desarrollo` del equipo. Conserva sus cinco
variables, el modelo inicial y las siete configuraciones de partida. Se añade
una comparación de anchura, semillas reproducibles, una prueba final y métricas
generadas automáticamente. La copia original está en `redes neuronales/originales/desarrollo/`.
