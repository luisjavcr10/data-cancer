import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, 
    GlobalAveragePooling3D, Dense, Dropout,
    BatchNormalization, concatenate
)
from tensorflow.keras.optimizers import Adam, AdamW
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize

def load_models():
    """Carga todos los modelos disponibles"""
    models = {}
    input_shape = (128, 128, 128, 1)  # Tamaño típico para volúmenes CT
    
    # Modelo 3D ResNet50
    base_model = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=input_shape
    )
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    resnet3d = Model(inputs=base_model.input, outputs=predictions)
    resnet3d.compile(
        optimizer=AdamW(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    models['3D ResNet50'] = resnet3d
    
    # Modelo 3D DenseNet121
    base_model = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=input_shape
    )
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    densenet3d = Model(inputs=base_model.input, outputs=predictions)
    densenet3d.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    models['3D DenseNet121'] = densenet3d
    
    # CNN 3D Personalizada
    inputs = Input(input_shape)
    x = Conv3D(32, (3,3,3), activation='relu')(inputs)
    x = MaxPooling3D((2,2,2))(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(64, (3,3,3), activation='relu')(x)
    x = MaxPooling3D((2,2,2))(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(128, (3,3,3), activation='relu')(x)
    x = MaxPooling3D((2,2,2))(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    custom_cnn = Model(inputs=inputs, outputs=outputs)
    custom_cnn.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    models['CNN 3D Personalizada'] = custom_cnn
    
    # Cargar pesos preentrenados si existen
    for name in models:
        model_path = f'models/{name.lower().replace(" ", "_")}_lung_cancer.h5'
        try:
            models[name] = load_model(model_path)
        except:
            print(f"No se encontraron pesos preentrenados para {name}")
    
    return models

def predict_volume(model, volume):
    """Realiza predicción en un volumen preprocesado"""
    # Asegurar que el volumen tenga la forma correcta
    if len(volume.shape) == 3:
        volume = np.expand_dims(volume, axis=-1)
    volume = np.expand_dims(volume, axis=0)
    
    # Realizar predicción
    prediction = model.predict(volume)
    confidence = np.max(prediction)
    diagnosis = 1 if prediction > 0.5 else 0  # 1: Positivo, 0: Negativo
    
    # Generar mapa de calor (saliency map)
    heatmap = generate_saliency_map(model, volume[0])
    
    return diagnosis, float(confidence), heatmap

def generate_saliency_map(model, volume):
    """Genera un mapa de saliencia para el volumen"""
    volume_tensor = tf.convert_to_tensor(np.expand_dims(volume, axis=0))
    
    with tf.GradientTape() as tape:
        tape.watch(volume_tensor)
        prediction = model(volume_tensor)
    
    # Calcular gradientes
    gradients = tape.gradient(prediction, volume_tensor)
    
    # Procesar gradientes para obtener mapa de saliencia
    saliency_map = np.max(np.abs(gradients.numpy()), axis=-1)[0]
    saliency_map = np.squeeze(saliency_map)
    
    # Normalizar y redimensionar
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    saliency_map = resize(saliency_map, volume.shape[:3], order=1, mode='reflect', anti_aliasing=True)
    
    return saliency_map