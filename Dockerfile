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
# y 'wget' para descargar el modelo de spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python (incluyendo spacy==3.4.4)
RUN pip install --no-cache-dir -r requirements.txt

# Descarga los datos de NLTK durante la construcción de la imagen
# Esto asegura que estén disponibles y no se descarguen en cada arranque
ENV NLTK_DATA /app/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('punkt', download_dir='/app/nltk_data')"

# --- INSTALACIÓN ULTRA-ROBUSTA DEL MODELO DE SPACY ---
# Descarga el archivo .whl del modelo es_core_news_sm compatible con spaCy 3.4.x
# Lo descargamos a /tmp para una ubicación limpia y conocida
RUN wget https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.4.0/es_core_news_sm-3.4.0-py3-none-any.whl -O /tmp/es_core_news_sm-3.4.0.whl

# Instala el modelo desde el archivo .whl descargado, usando la ruta completa y explícita
# Esto le dice a pip que es un archivo local y no un nombre de paquete
RUN pip install /tmp/es_core_news_sm-3.4.0.whl

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
