# Binary_classification_NLP

Este repositorio contiene un modelo de clasificación binaria utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) con el conjunto de datos IMDB. El modelo está construido con TensorFlow y Keras para clasificar reseñas de películas como positivas o negativas.

## Estructura del Proyecto
```
├── src/
│   ├── Bin_Class_NLP_code.py  # Contiene la implementación del modelo de clasificación binaria
├── main.py  # Punto de entrada para ejecutar el modelo
├── requirements.txt  # Dependencias necesarias para ejecutar el proyecto
├── README.md  # Documentación del proyecto
├── .gitignore  # Especifica los archivos y carpetas que deben ser ignorados por Git
```

## Instalación
Para ejecutar este proyecto, necesitas tener Python instalado. Luego, sigue estos pasos:

1. **Clonar el repositorio:**
   ```sh
   git clone https://github.com/Hecedu09/Binary_classification_NLP_HectorRamirez
   cd Binary_classification_NLP_HectorRamirez
   ```

2. **Crear un entorno virtual (opcional pero recomendado):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # En macOS/Linux
   venv\Scripts\activate  # En Windows
   ```

3. **Instalar las librerías necesarias:**
   ```sh
   pip install -r requirements.txt
   ```

### Librerías Utilizadas
El proyecto utiliza las siguientes librerías:
- `numpy`: Para operaciones numéricas.
- `matplotlib`: Para visualizar los resultados del entrenamiento y validación.
- `tensorflow` & `keras`: Para construir y entrenar la red neuronal.
- `graphviz`: Necesaria para la visualización del modelo.

### Instalación de Graphviz
La librería `graphviz` es necesaria para que `plot_model()` funcione correctamente. Instálala con:
   ```sh
   pip install graphviz
   ```
Además, es necesario instalar el paquete del sistema Graphviz:
- **Windows:** Descarga e instala desde [Graphviz Official Site](https://graphviz.gitlab.io/download/)
- **Linux/macOS:**
   ```sh
   sudo apt install graphviz  # Ubuntu/Debian
   brew install graphviz  # macOS
   ```

## Cómo Usar
1. Ejecuta el script `main.py` para iniciar el modelo de clasificación binaria:
   ```sh
   python main.py
   ```

2. El modelo se entrenará utilizando el conjunto de datos IMDB y generará gráficos de precisión y pérdida.

3. Después del entrenamiento, el modelo evaluará los datos de prueba y realizará predicciones.