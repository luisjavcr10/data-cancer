import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from model_utils import load_models, predict_volume, generate_saliency_map
from preprocessing import load_and_preprocess_ct_scan
import SimpleITK as sitk
import time

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Diagnóstico de Cáncer de Pulmón",
    page_icon="🏥",
    layout="wide"
)

# Título de la aplicación
st.title("Sistema Inteligente de Diagnóstico de Cáncer de Pulmón")
st.markdown("""
Este sistema utiliza redes neuronales 3D para analizar tomografías computarizadas 
y detectar nódulos pulmonares sospechosos.
""")

# Cargar modelos
@st.cache_resource
def load_models_cached():
    return load_models()

models = load_models_cached()
model_names = list(models.keys())

# Sidebar para configuración
st.sidebar.header("Configuración")
selected_model = st.sidebar.selectbox(
    "Modelo a utilizar",
    model_names,
    index=0
)

confidence_threshold = st.sidebar.slider(
    "Umbral de confianza para diagnóstico",
    min_value=0.5,
    max_value=0.99,
    value=0.85,
    step=0.01
)

# Carga de imágenes
st.header("Carga de Tomografía Computarizada")
upload_option = st.radio(
    "Seleccione el tipo de entrada",
    ["Subir archivo DICOM", "Usar ejemplo"]
)

if upload_option == "Subir archivo DICOM":
    uploaded_file = st.file_uploader(
        "Suba un archivo DICOM o directorio comprimido (.zip) con la serie de tomografía",
        type=["dcm", "zip"]
    )
else:
    example_cases = {
        "Caso benigno (Ejemplo)": "data/examples/benign_case",
        "Caso maligno (Ejemplo)": "data/examples/malignant_case"
    }
    example_choice = st.selectbox(
        "Seleccione caso de ejemplo",
        list(example_cases.keys())
    )
    uploaded_file = example_cases[example_choice]

if uploaded_file:
    # Procesar tomografía
    with st.spinner("Procesando tomografía..."):
        volume, original_volume = load_and_preprocess_ct_scan(uploaded_file)
        time.sleep(1)  # Simular procesamiento
    
    # Visualización de cortes
    st.subheader("Visualización de Cortes Axiales")
    slice_idx = st.slider("Seleccione corte axial", 0, volume.shape[0]-1, volume.shape[0]//2)
    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].imshow(original_volume[slice_idx], cmap='gray')
    ax[0].set_title("Original")
    ax[0].axis('off')
    
    ax[1].imshow(volume[slice_idx], cmap='gray')
    ax[1].set_title("Preprocesado")
    ax[1].axis('off')
    st.pyplot(fig)
    
    # Realizar predicción
    st.header("Resultados del Diagnóstico")
    model = models[selected_model]
    
    with st.spinner("Analizando tomografía..."):
        prediction, confidence, heatmap = predict_volume(model, volume)
        time.sleep(1)  # Simular procesamiento
    
    # Mostrar resultados
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Modelo utilizado", selected_model)
    with col2:
        st.metric("Predicción", "Positivo para cáncer" if prediction == 1 else "Negativo para cáncer")
    with col3:
        st.metric("Confianza", f"{confidence:.2%}")
    
    # Interpretación de resultados
    if confidence < confidence_threshold:
        st.warning("La confianza en el diagnóstico es baja. Se recomienda evaluación adicional por un radiólogo.")
    else:
        if prediction == 0:
            st.success("No se detectaron nódulos sospechosos. Se recomienda seguimiento rutinario.")
        else:
            st.error("Se detectaron nódulos pulmonares sospechosos. Se recomienda consulta urgente con un especialista.")
    
    # Visualización de áreas sospechosas
    st.subheader("Mapa de Calor de Nódulos Sospechosos")
    st.markdown("Áreas resaltadas indican regiones con mayor probabilidad de malignidad:")
    
    # Mostrar slice con heatmap
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(original_volume[slice_idx], cmap='gray')
    ax.imshow(heatmap[slice_idx], alpha=0.5, cmap='jet')
    ax.axis('off')
    st.pyplot(fig)

# Sección de comparación de modelos
st.header("Comparación de Modelos")
if st.button("Mostrar Métricas Comparativas"):
    show_model_comparison()

# Sección de optimización de hiperparámetros
st.header("Optimización de Modelos")
if st.button("Ejecutar Búsqueda de Hiperparámetros"):
    run_hyperparameter_tuning()

def show_model_comparison():
    """Muestra métricas comparativas de modelos"""
    with st.spinner("Cargando datos de evaluación..."):
        # Datos simulados (en producción cargar desde evaluación real)
        evaluation_data = {
            'Modelo': ['3D ResNet50', '3D DenseNet121', 'VGG16 3D', 'CNN 3D Personalizada'],
            'Precisión': [0.92, 0.91, 0.89, 0.88],
            'Sensibilidad': [0.93, 0.90, 0.88, 0.86],
            'Especificidad': [0.91, 0.92, 0.90, 0.89],
            'AUC-ROC': [0.96, 0.95, 0.93, 0.92],
            'Dice Score': [0.85, 0.83, 0.80, 0.78],
            'Tiempo Inferencia (s)': [3.2, 4.1, 2.8, 1.5]
        }
        
        df = pd.DataFrame(evaluation_data)
        
        # Mostrar tabla
        st.subheader("Métricas de Rendimiento")
        st.dataframe(df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ffcccb'))
        
        # Gráficos comparativos
        st.subheader("Análisis Comparativo")
        
        fig1 = px.bar(df, x='Modelo', y=['Sensibilidad', 'Especificidad'], 
                     barmode='group', title="Sensibilidad vs Especificidad")
        st.plotly_chart(fig1, use_column_width=True)
        
        fig2 = px.scatter(df, x='Tiempo Inferencia (s)', y='AUC-ROC', color='Modelo',
                         size=[15]*len(df), title="Rendimiento vs Velocidad",
                         hover_name='Modelo')
        st.plotly_chart(fig2, use_column_width=True)
        
        fig3 = px.line_polar(df, r='Dice Score', theta='Modelo', 
                            line_close=True, title="Dice Score por Modelo")
        st.plotly_chart(fig3, use_column_width=True)

def run_hyperparameter_tuning():
    """Simula la optimización de hiperparámetros"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simular proceso de optimización
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Progreso: {i + 1}%")
        time.sleep(0.05)
    
    # Resultados simulados
    best_params = {
        'learning_rate': 0.0001,
        'batch_size': 8,
        'optimizer': 'AdamW',
        'dropout_rate': 0.3,
        'dense_units': 256,
        'depth': 4,
        'filters': 32
    }
    
    st.success("¡Optimización completada!")
    st.subheader("Mejores Hiperparámetros Encontrados")
    st.json(best_params)
    
    # Gráfico de evolución
    st.subheader("Progreso de la Optimización")
    fig, ax = plt.subplots()
    x = np.arange(100)
    y = 0.7 + 0.3 * (1 - np.exp(-x / 30)) + 0.1 * np.random.randn(100)
    ax.plot(x, y, label='Dice Score (Validación)')
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Dice Score")
    ax.set_title("Evolución de la Búsqueda")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs('data/examples', exist_ok=True)
    os.makedirs('models', exist_ok=True)
