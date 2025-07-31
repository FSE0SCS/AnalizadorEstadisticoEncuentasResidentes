# Usa una imagen base de Python más completa y estable para asegurar las herramientas de compilación
FROM python:3.9

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requisitos y el archivo de base de datos primero para aprovechar el caché de Docker
COPY requirements.txt .
COPY prompt_data.db .

# Actualiza pip, setuptools y wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala las dependencias del sistema necesarias para compilar ciertas librerías (como thinc/spaCy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    # Ya no necesitamos wget aquí, pero lo mantendremos por si acaso en futuras necesidades
    # wget \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python (incluyendo spacy==3.4.4)
RUN pip install --no-cache-dir -r requirements.txt

# Descarga los datos de NLTK durante la construcción de la imagen
# Esto asegura que estén disponibles y no se descarguen en cada arranque
ENV NLTK_DATA /app/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('punkt', download_dir='/app/nltk_data')"

# --- INSTALACIÓN DIRECTA DEL MODELO DE SPACY DESDE LA URL (NUEVO ENFOQUE) ---
# Pip puede instalar directamente desde una URL de un archivo .whl
RUN pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.4.0/es_core_news_sm-3.4.0-py3-none-any.whl

# La línea `spacy link` no es estrictamente necesaria si se instala el .whl directamente
# y spaCy lo reconoce automáticamente. Si la aplicación no carga el modelo, la reintroduciremos.
# RUN python -m spacy link es_core_news_sm es_core_news_sm --force

# Copia el resto de los archivos de tu aplicación al contenedor
COPY . .

# Expone el puerto que Streamlit usará
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit cuando el contenedor se inicie
# `--server.port` y `--server.enableCORS` son importantes para el despliegue web
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]
