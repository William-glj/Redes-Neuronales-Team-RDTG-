# Revisión a partir de desarrollo

Base revisada: rama `desarrollo`, commit `0857c4b05a69e11ffdf3df62ee57eaf9d91ebad2`.
Se conserva una copia íntegra del ZIP de esa revisión en `originales/desarrollo/`.

## Qué se conserva

- Cine, CSV de partida y clasificación de `popularity_level`.
- Umbral comercial de presupuesto y recaudación superiores a 10.000.
- Cinco variables del equipo: presupuesto, duración, votos, nota media y año.
- Nombres de los tres scripts de preparación/entrenamiento y `presentacion.qmd`.
- Modelo inicial de dos capas de 64, lote 32, tasa 0,01 y diez épocas.
- Las siete configuraciones originales de capas, lotes y tasas, con 15 épocas.

## Qué se corrige o añade

1. Se elimina la etiqueta ausente antes de codificar las clases. `astype(str)`
   la convertía en `"nan"`, que podía entrar como una quinta categoría.
2. Se explicita el orden Baja, Media, Media-alta y Alta. El orden alfabético
   de LabelEncoder no coincidía con algunos ejemplos explicados en el informe.
3. Se distingue selección comercial de dato incorrecto. El umbral excluye
   películas que pueden ser válidas y limita la representatividad.
4. Se añaden estadísticas de nulos, IQR y cinco figuras del EDA.
5. Presupuesto y votos se transforman con logaritmo antes de StandardScaler.
   Medianas y escalado se ajustan exclusivamente con entrenamiento.
6. Se separan entrenamiento, validación y prueba final, manteniendo las
   proporciones de clases. El modelo inicial se evalúa solo en validación.
7. Se añade Exp_8, una capa de 64, para comparar su anchura con Exp_1, de 32.
   Las ocho configuraciones se repiten con tres semillas: 24 entrenamientos.
8. Se selecciona por F1 macro medio de validación y se recupera la época de
   menor pérdida de validación. Prueba se consulta después de la elección.
9. Se guardan resultados, predicciones y pesos. Las gráficas y tablas se leen
   de esas salidas; no se mantienen las cifras escritas a mano de la versión anterior.
10. Se corrigen ejemplos incompatibles con el periodo del CSV y afirmaciones
    que daban por demostrada la superioridad de una red sin compararla.
11. Se añade una guía oral para cuatro personas y se organiza la entrega dentro
    de `redes neuronales`, como exige el enunciado.

Los porcentajes de esta revisión no se comparan directamente con los de la
presentación anterior: cambian el reparto y la preparación de los datos.
Tampoco deben mezclarse con el primer ensayo local basado en la copia de main.

## Pendiente de confirmar por el equipo

- Fuente original exacta del CSV y creación de las etiquetas.
- Registro de integrantes/repositorio en Loop.
- Publicación y comprobación de la URL final; exposición de los cuatro integrantes.

La rama de destino para la subida es `desarrollo` en minúsculas.
