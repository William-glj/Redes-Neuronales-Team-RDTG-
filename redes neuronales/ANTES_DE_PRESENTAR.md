# Preparación para el 23 de octubre

## Qué abrir

Para exponer el trabajo ya calculado, abrir **`presentacion.html`** desde esta
carpeta con el navegador. Es el informe Quarto guardado con imágenes y recursos
incrustados. No necesita instalar Python ni ejecutar entrenamientos para leerlo.
El archivo `_site/index.html` es la salida del último render; después de
regenerarlo hay que actualizar la copia `presentacion.html`.

Para preparar las intervenciones, abrir `GUIA_EXPOSICION.md`. Dylan tiene una
explicación más pausada en `EXPLICACION_DYLAN.md`.
`EXPLICACION_COMPLETA.md` recorre todo el proyecto desde cero. Para estudiar
con explicaciones visuales, abrir `repaso_proyecto.html` en el navegador y
pulsar **Reproducir**. El menú permite ir directamente a cada capítulo.

## Esta tarde

1. Repartir las cuatro partes de la guía y hacer un ensayo con cronómetro:
   unos 8–10 minutos, ajustándolo al tiempo que indique el profesor.
2. Cada persona debe explicar al menos un gráfico o fragmento de código.
   Ensayar las transiciones entre compañeros.
3. Todos deben entender objetivo, cinco entradas, tres conjuntos y resultado.
4. Guardar `presentacion.html` en el ordenador que se utilizará y en una
   segunda ubicación accesible. Abrirlo allí para comprobar que se ve.

## Publicación pendiente

El commit `2f5399d` se subió correctamente a `desarrollo`. En Actions terminaron
bien el EDA, los entrenamientos y el render. La publicación quedó bloqueada
por las reglas del entorno `github-pages`, que solo admiten `gh-pages`.

Un administrador del repositorio debe seleccionar GitHub Actions como fuente
de Pages y permitir `desarrollo` en el entorno `github-pages`.
La cuenta colaboradora consultada tiene WRITE; esos ajustes requieren administración.
Esta revisión incluye la fuente contrastada, las fechas corregidas y el material de repaso.

Después de completar el despliegue de esta revisión, abrir la web desde
otro dispositivo y verificar que aparece la fuente de Kaggle contrastada,
la corrección de las 12 fechas y la tabla de ejemplos de películas. La URL
del repositorio y la URL del informe son enlaces distintos. La copia local
sirve para exponer si hay un problema de conexión; la entrega exige URL pública.

Comprobar también el registro de integrantes y repositorio en Loop, y publicar
allí la URL del informe antes del límite. No dar esos pasos por realizados sin comprobarlos.

## Antes de ser el primer grupo

- Llevar el informe abierto, el cargador y el adaptador de proyector que necesitéis.
- Probar la conexión al proyector y ajustar el zoom para leer las tablas.
- Situarse en la introducción y cerrar las notificaciones que puedan interrumpir.
- No instalar herramientas ni volver a entrenar durante la defensa.
- Si preguntan una causa que no habéis investigado, explicar qué muestra el
  gráfico y qué no permite concluir. No inventar una explicación.

## Frases que debemos evitar

| Evitar | Explicación correcta |
| --- | --- |
| «Predecimos el éxito antes del estreno». | Clasificamos popularidad con datos posteriores al estreno. |
| «La red es mucho mejor». | En esta prueba mejora a la logística unos 2,1 puntos de accuracy; la diferencia es pequeña. |
| «Todos los atípicos eran errores». | El IQR señala extremos; conservamos los que pueden ser válidos y transformamos escalas. |
| «F1 0,633 significa 63,3 % de aciertos». | F1 y accuracy son métricas distintas. |
| «Hicimos validación cruzada». | Repetimos tres semillas sobre una única partición. |
| «Ya está entregado porque está en GitHub». | Hay que publicar el informe y poner su URL en Loop. |
