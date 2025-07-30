# Usa una imagen base de Python. Python 3.11 es una buena opción estable.
# Asegúrate de que la versión de Python aquí sea compatible con tus librerías.
FROM python:3.11-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requisitos y el archivo de base de datos primero para aprovechar el caché de Docker
COPY requirements.txt .
COPY prompt_data.db .

# Instala las dependencias de Python
# `--no-cache-dir` para no guardar el caché de pip, reduciendo el tamaño de la imagen
# `uv` es el nuevo instalador de Streamlit, pero pip es más universal para Dockerfiles
RUN pip install --no-cache-dir -r requirements.txt

# Descarga los datos de NLTK durante la construcción de la imagen
# Esto asegura que estén disponibles y no se descarguen en cada arranque
ENV NLTK_DATA /app/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('punkt', download_dir='/app/nltk_data')"

# Descarga el modelo de spaCy.
# 'es_core_news_sm' ya está en requirements.txt, pero esta línea asegura su instalación
# y que los datos asociados estén disponibles para spaCy.
# Si ya está en requirements.txt, pip lo instalará, pero esta línea asegura que spacy lo "reconozca"
# y lo configure internamente si es necesario.
# Sin embargo, la forma más limpia es que `pip install es_core_news_sm` ya lo haga.
# Si `pip install es_core_news_sm` no instala los datos, entonces necesitaríamos `spacy download es_core_news_sm`.
# Pero como lo hemos añadido a requirements.txt, pip debería manejarlo.
# Si el error persiste, descomentar la siguiente línea:
# RUN python -m spacy download es_core_news_sm

# Copia el resto de los archivos de tu aplicación al contenedor
COPY . .

# Expone el puerto que Streamlit usará
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit cuando el contenedor se inicie
# `--server.port` y `--server.enableCORS` son importantes para el despliegue web
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]
