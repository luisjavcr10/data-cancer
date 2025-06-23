import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from model_utils import load_models
from preprocessing import load_and_preprocess_ct_scan
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_dataset(data_dir, test_size=0.2, random_state=42):
    """Carga y divide el dataset en entrenamiento y validación"""
    # Obtener listas de casos benignos y malignos
    benign_cases = [os.path.join(data_dir, 'benign', f) for f in os.listdir(os.path.join(data_dir, 'benign'))]
    malignant_cases = [os.path.join(data_dir, 'malignant', f) for f in os.listdir(os.path.join(data_dir, 'malignant'))]
    
    # Crear etiquetas
    benign_labels = [0] * len(benign_cases)
    malignant_labels = [1] * len(malignant_cases)
    
    # Combinar y dividir
    all_cases = benign_cases + malignant_cases
    all_labels = benign_labels + malignant_labels
    
    # Dividir en train y test
    train_cases, val_cases, train_labels, val_labels = train_test_split(
        all_cases, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    return train_cases, val_cases, train_labels, val_labels

class CTScanGenerator(tf.keras.utils.Sequence):
    """Generador de datos para volúmenes CT"""
    def __init__(self, cases, labels, batch_size=8, target_size=(128, 128, 128)):
        self.cases = cases
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(self.cases))
    
    def __len__(self):
        return int(np.ceil(len(self.cases) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_cases = [self.cases[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        # Cargar y preprocesar volúmenes
        batch_volumes = []
        for case in batch_cases:
            volume, _ = load_and_preprocess_ct_scan(case, self.target_size)
            batch_volumes.append(np.expand_dims(volume, axis=-1))
        
        return np.array(batch_volumes), np.array(batch_labels)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def train_model(model, train_generator, val_generator, epochs=100, model_name='lung_cancer_model'):
    """Entrena un modelo con los generadores de datos"""
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'models/{model_name}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Entrenamiento
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar historial
    pd.DataFrame(history.history).to_csv(f'models/{model_name}_history.csv', index=False)
    
    return history

if __name__ == "__main__":
    # Configuración
    data_dir = 'data/examples'  # Directorio con dataset LIDC-IDRI
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Cargar dataset
    train_cases, val_cases, train_labels, val_labels = load_dataset(data_dir)
    
    # Crear generadores
    train_gen = CTScanGenerator(train_cases, train_labels, batch_size=1)
    val_gen = CTScanGenerator(val_cases, val_labels, batch_size=1)
    
    # Cargar y entrenar modelos
    models = load_models()
    for name, model in models.items():
        print(f"\nEntrenando modelo: {name}")
        history = train_model(
            model,
            train_gen,
            val_gen,
            epochs=100,
            model_name=f"{name.lower().replace(' ', '_')}_lung_cancer"
        )