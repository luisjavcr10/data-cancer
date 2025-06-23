# Usar imagen base de TensorFlow optimizada para CPU
FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# Instalar dependencias del sistema optimizadas para Mac
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-opencv \
    git \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Variables de entorno para optimización CPU
ENV OMP_NUM_THREADS=4
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar MONAI para procesamiento médico (versión CPU)
RUN pip install monai==0.9.1

# Instalar dependencias adicionales para mejor rendimiento en CPU
RUN pip install --no-cache-dir \
    openvino-dev==2022.3.0 \
    intel-tensorflow==2.10.0

# Copiar aplicación
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]