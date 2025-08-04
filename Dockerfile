# Dockerfile
# Este archivo define la imagen de Docker para la aplicación de Streamlit.

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
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python, especificando el índice extra para PyTorch.
# Esto es CRUCIAL para que pip encuentre las versiones +cpu de torch.
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Descarga los datos de NLTK durante la construcción de la imagen.
# Esto asegura que estén disponibles y no se descarguen en cada arranque, lo que acelera el inicio de la app.
ENV NLTK_DATA /app/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('punkt', download_dir='/app/nltk_data')"

# INSTALACIÓN DIRECTA DEL MODELO DE SPACY DESDE LA URL.
# Esta es una forma robusta de asegurar que el modelo correcto esté disponible.
RUN pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.4.0/es_core_news_sm-3.4.0-py3-none-any.whl

# Copia el resto de los archivos de tu aplicación al contenedor
COPY . .

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación de Streamlit cuando el contenedor se inicie
CMD ["streamlit", "run", "app.py"]
