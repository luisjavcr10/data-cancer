import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
from train import CTScanGenerator, load_dataset
import os

def build_model(hp):
    """Construye modelo con hiperparámetros a optimizar"""
    # Hiperparámetros a optimizar
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.6, step=0.1)
    dense_units = hp.Int('dense_units', min_value=64, max_value=512, step=64)
    optimizer = hp.Choice('optimizer', ['adam', 'adamw', 'rmsprop'])
    num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=5)
    filters_base = hp.Int('filters_base', min_value=16, max_value=64, step=16)
    
    # Arquitectura del modelo
    inputs = tf.keras.Input(shape=(128, 128, 128, 1))
    x = inputs
    
    # Capas convolucionales
    for i in range(num_conv_layers):
        filters = filters_base * (2 ** i)
        x = layers.Conv3D(filters, (3,3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2,2,2))(x)
    
    # Capas fully connected
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Optimizador
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'adamw':
        opt = AdamW(learning_rate=learning_rate)
    else:
        opt = RMSprop(learning_rate=learning_rate)
    
    # Compilar modelo
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def tune_hyperparameters(data_dir, max_epochs=50, executions_per_trial=1):
    """Ejecuta la optimización de hiperparámetros"""
    # Configurar el tuner
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=max_epochs,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='lung_cancer_detection',
        overwrite=True
    )
    
    # Cargar datos
    train_cases, val_cases, train_labels, val_labels = load_dataset(data_dir)
    
    # Crear generadores
    train_gen = CTScanGenerator(train_cases, train_labels, batch_size=4)
    val_gen = CTScanGenerator(val_cases, val_labels, batch_size=4)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Búsqueda
    tuner.search(
        train_gen,
        validation_data=val_gen,
        epochs=max_epochs,
        callbacks=callbacks
    )
    
    # Obtener mejores hiperparámetros
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Entrenar modelo final con mejores hiperparámetros
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=max_epochs,
        callbacks=callbacks
    )
    
    # Guardar mejor modelo
    model.save('models/optimized_lung_cancer_model.h5')
    
    return best_hps, history

if __name__ == "__main__":
    data_dir = 'data/LIDC-IDRI'  # Directorio con datos de entrenamiento
    os.makedirs('hyperparameter_tuning', exist_ok=True)
    
    print("Iniciando búsqueda de hiperparámetros...")
    best_hyperparameters, history = tune_hyperparameters(data_dir)
    
    print("\nMejores hiperparámetros encontrados:")
    for param, value in best_hyperparameters.values.items():
        print(f"{param}: {value}")
    
    # Guardar historial
    pd.DataFrame(history.history).to_csv(
        'hyperparameter_tuning/optimization_history.csv',
        index=False
    )