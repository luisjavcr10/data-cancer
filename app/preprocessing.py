import numpy as np
import SimpleITK as sitk
import pydicom
import os
import zipfile
from skimage.transform import resize
from skimage.exposure import equalize_adapthist

def load_dicom_series(directory):
    """Carga una serie DICOM desde un directorio"""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    
    # Convertir a array numpy
    volume = sitk.GetArrayFromImage(image)
    
    # Reordenar ejes (SimpleITK usa z,y,x)
    volume = np.transpose(volume, (2, 1, 0))
    
    return volume, image.GetSpacing()

def load_and_preprocess_ct_scan(input_path, target_size=(128, 128, 128)):
    """
    Carga y preprocesa un volumen de TC
    Args:
        input_path: Ruta al archivo DICOM, directorio o zip
        target_size: Tamaño objetivo para el volumen
    Returns:
        volume_preprocessed: Volumen preprocesado
        original_volume: Volumen original (redimensionado)
    """
    # Cargar datos según el tipo de entrada
    if isinstance(input_path, str) and input_path.endswith('.zip'):
        # Extraer archivo zip
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            temp_dir = 'temp_dicom'
            zip_ref.extractall(temp_dir)
            input_path = temp_dir
    
    if os.path.isdir(input_path):
        # Cargar serie DICOM
        volume, spacing = load_dicom_series(input_path)
    else:
        # Cargar archivo DICOM individual (no recomendado)
        ds = pydicom.dcmread(input_path)
        volume = ds.pixel_array
        spacing = (1.0, 1.0, 1.0)  # Asumir si no hay información
    
    # Guardar volumen original (para visualización)
    original_volume = resize(volume, target_size, mode='reflect', anti_aliasing=True)
    
    # Preprocesamiento
    volume_preprocessed = preprocess_volume(volume, spacing, target_size)
    
    return volume_preprocessed, original_volume

def preprocess_volume(volume, spacing, target_size):
    """Aplica preprocesamiento a un volumen CT"""
    # 1. Normalizar intensidades (ventana pulmonar)
    volume = apply_lung_window(volume)
    
    # 2. Resample para tamaño de voxel isotrópico
    volume = resample_volume(volume, spacing)
    
    # 3. Redimensionar al tamaño objetivo
    volume = resize(volume, target_size, mode='reflect', anti_aliasing=True)
    
    # 4. Ecualización adaptativa de histograma (slice por slice)
    for i in range(volume.shape[0]):
        volume[i] = equalize_adapthist(volume[i])
    
    # 5. Normalización [0,1]
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    return volume

def apply_lung_window(volume, level=-600, width=1500):
    """Aplica ventana pulmonar a un volumen CT"""
    window_min = level - width / 2
    window_max = level + width / 2
    
    volume = np.clip(volume, window_min, window_max)
    volume = (volume - window_min) / (window_max - window_min)
    
    return volume

def resample_volume(volume, original_spacing, target_spacing=1.0):
    """Re-muestrea el volumen a un espaciado isotrópico"""
    # Calcular factores de resample
    resize_factor = [os * target_spacing for os in original_spacing]
    
    # Calcular nueva forma del volumen
    new_shape = (
        int(volume.shape[0] * resize_factor[0]),
        int(volume.shape[1] * resize_factor[1]),
        int(volume.shape[2] * resize_factor[2])
    )
    
    # Redimensionar volumen
    volume = resize(
        volume,
        new_shape,
        mode='reflect',
        anti_aliasing=True
    )
    
    return volume