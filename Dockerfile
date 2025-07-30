# Usa una imagen base de Python más completa para asegurar las herramientas de compilación
FROM python:3.11

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requisitos y el archivo de base de datos primero para aprovechar el caché de Docker
COPY requirements.txt .
COPY prompt_data.db .

# Actualiza pip, setuptools y wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala las dependencias de Python
# La imagen base "python:3.11" ya incluye la mayoría de las herramientas de construcción,
# por lo que la línea `apt-get install` anterior podría no ser estrictamente necesaria ahora,
# pero la mantendremos por si acaso.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     g++ \
#     build-essential \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Descarga los datos de NLTK durante la construcción de la imagen
# Esto asegura que estén disponibles y no se descarguen en cada arranque
ENV NLTK_DATA /app/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('punkt', download_dir='/app/nltk_data')"

# Descarga el modelo de spaCy.
# 'es_core_news_sm' ya está en requirements.txt, por lo que pip lo instalará.
# No es necesario un comando `spacy download` adicional si pip lo maneja.

# Copia el resto de los archivos de tu aplicación al contenedor
COPY . .

# Expone el puerto que Streamlit usará
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit cuando el contenedor se inicie
# `--server.port` y `--server.enableCORS` son importantes para el despliegue web
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]
