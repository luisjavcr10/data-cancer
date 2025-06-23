import os
# Configuración para optimizar rendimiento en CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'

import tensorflow as tf
# Configurar TensorFlow para CPU optimizado
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, 
    GlobalAveragePooling3D, Dense, Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam, AdamW
import numpy as np
from skimage.transform import resize

def load_models():
    """Carga todos los modelos disponibles y sus pesos preentrenados si existen."""
    models = {}
    input_shape = (128, 128, 128, 1)  # Tamaño estándar de entrada

    # Modelo 3D ResNet50
    base_resnet = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=input_shape
    )
    x = GlobalAveragePooling3D()(base_resnet.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    resnet3d = Model(inputs=base_resnet.input, outputs=output, name='ResNet3D')
    resnet3d.compile(
        optimizer=AdamW(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    models['3D ResNet50'] = resnet3d

    # Modelo 3D DenseNet121
    base_densenet = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=input_shape
    )
    x = GlobalAveragePooling3D()(base_densenet.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    densenet3d = Model(inputs=base_densenet.input, outputs=output, name='DenseNet3D')
    densenet3d.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    models['3D DenseNet121'] = densenet3d

    # Modelo CNN 3D personalizado
    inputs = Input(shape=input_shape, name="input_volume")
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    custom_cnn = Model(inputs=inputs, outputs=output, name='Custom3DCNN')
    custom_cnn.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    models['CNN 3D Personalizada'] = custom_cnn

    # Cargar pesos si existen
    for name, model in models.items():
        model_path = f'models/{name.lower().replace(" ", "_")}_lung_cancer.h5'
        try:
            models[name] = load_model(model_path)
        except (OSError, IOError):
            print(f"[AVISO] No se encontraron pesos preentrenados para: {name}. Se usará el modelo sin pesos.")

    return models


def predict_volume(model, volume):
    """
    Realiza una predicción binaria en un volumen 3D.
    Devuelve el diagnóstico, confianza y un mapa de saliencia.
    """
    # Asegurar formato (1, D, H, W, 1)
    if volume.ndim == 3:
        volume = volume[..., np.newaxis]
    volume = np.expand_dims(volume, axis=0)

    # Predicción
    prediction = model.predict(volume)
    confidence = float(prediction[0][0])
    diagnosis = int(confidence > 0.5)

    # Mapa de saliencia
    heatmap = generate_saliency_map(model, volume[0])

    return diagnosis, confidence, heatmap


def generate_saliency_map(model, volume):
    """
    Genera un mapa de saliencia basado en gradientes absolutos.
    """
    volume_tensor = tf.convert_to_tensor(np.expand_dims(volume, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(volume_tensor)
        prediction = model(volume_tensor)

    gradients = tape.gradient(prediction, volume_tensor)
    gradients_np = gradients.numpy()[0]

    # Saliency map por magnitud máxima del gradiente
    saliency_map = np.max(np.abs(gradients_np), axis=-1)

    # Normalización defensiva
    min_val, max_val = saliency_map.min(), saliency_map.max()
    if max_val - min_val != 0:
        saliency_map = (saliency_map - min_val) / (max_val - min_val)
    else:
        saliency_map = np.zeros_like(saliency_map)

    # Redimensionar a forma original
    saliency_map = resize(saliency_map, volume.shape[:3], order=1, mode='reflect', anti_aliasing=True)

    return saliency_map
