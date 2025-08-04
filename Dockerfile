# Usa una imagen base de Python más reciente y estable
FROM python:3.10

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requisitos y el archivo de base de datos primero para aprovechar el caché de Docker
COPY requirements.txt .
COPY prompt_data.db .

# Actualiza pip, setuptools y wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala las dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instala PyTorch de forma separada usando su índice de descarga
# Esto es CRUCIAL para evitar errores de resolución de dependencia con versiones +cpu
RUN pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Instala el resto de las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Descarga los datos de NLTK durante la construcción de la imagen
ENV NLTK_DATA /app/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('punkt', download_dir='/app/nltk_data')"

# INSTALACIÓN DIRECTA DEL MODELO DE SPACY DESDE LA URL
# Esto se mantiene igual para asegurar la versión correcta del modelo
RUN pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.4.0/es_core_news_sm-3.4.0-py3-none-any.whl

# Copia el resto de los archivos de tu aplicación al contenedor
COPY . .

# Expone el puerto que Streamlit usa por defecto
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py"]
