FROM tensorflow/tensorflow:2.10.0
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-opencv \
    git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install monai==0.9.1
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]