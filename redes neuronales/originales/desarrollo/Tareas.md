# Trabajo 1: Exploración y Entrenamiento de Redes Neuronales

## 1.Organización y Configuración Inicial 
(HECHO)

- **Formación de equipos:** La clase se dividirá en tres grupos (dos equipos de 4 personas y un equipo de 3 personas).

- **Temática del proyecto:** Cada equipo elegirá un tema genérico de su interés (deportes, cine, naturaleza, mascotas, salud, etc.). Importante: Este tema será el hilo conductor y se mantendrá para todos los  del curso.

- **Repositorio de GitHub:** Cada equipo creará un único repositorio en GitHub que centralizará todos los proyectos del curso. Para esta primera entrega, todo el código y los recursos deberán guardarse dentro de una carpeta específica llamada redes neuronales.

- **Registro de equipos:** Un representante de cada grupo deberá publicar en Loop (Teams) el nombre de los integrantes y la URL del repositorio de GitHub.


## 2. Fases del Desarrollo del Proyecto
### Fase A: Obtención de Datos y Análisis Exploratorio (EDA)
(HECHO)
- **Dataset:** Buscar y seleccionar un conjunto de datos público (utilizando plataformas como Kaggle, Hugging Face, etc.) que esté relacionado con la temática elegida por el equipo.

- **Exploración y limpieza:** Utilizando numpy y pandas, se realizará un EDA para descubrir las variables de interés. Se deben identificar, analizar y tratar adecuadamente los valores nulos y los datos atípicos (eliminándolos, imputándolos, etc.).

- **Visualización:** El EDA debe acompañarse de gráficos explicativos utilizando librerías como matplotlib, seaborn, plotnine o plotly.
//sin hacer gráficos

### Fase B: Definición y Entrenamiento del Modelo
(HECHO)

- **Planteamiento del problema:** Se debe elegir una variable dependiente de tipo categórico (problema de clasificación) y un conjunto de variables explicativas. El problema abordado debe tener la complejidad suficiente para justificar el uso de una red neuronal (no será válida una simple regresión).

- **Framework:** Para la construcción y el entrenamiento del modelo, es obligatorio utilizar PyTorch o PyTorch Lightning.

### Fase C: Experimentación y Análisis de Resultados

- **Ajuste de hiperparámetros:** No basta con entrenar un único modelo. Se deberán probar diferentes arquitecturas de red para evaluar cómo cambia el rendimiento al modificar parámetros como:
- Número de capas ocultas.
- Número de neuronas por capa.
- Tamaño del batch (batch size).
- Tasa de aprendizaje (learning rate).
- Conclusiones: Se analizarán y compararán los resultados obtenidos en las diferentes pruebas, justificando qué arquitectura funciona mejor y por qué.

## 3. Entregables y Documentación

- **Publicación en Quarto:** Todo el trabajo realizado (introducción, EDA, decisiones de diseño, experimentación y conclusiones) deberá estar documentado y publicado a través de Quarto.
- Código en el informe: El documento publicado debe incluir explicaciones en texto intercaladas con los fragmentos de código más relevantes del proyecto.
- **Entrega final:** La URL de la página web generada con Quarto deberá publicarse en Loop antes de la fecha límite.



