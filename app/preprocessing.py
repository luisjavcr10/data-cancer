import numpy as np
import SimpleITK as sitk
import pydicom
import os
import zipfile
import shutil
from skimage.transform import resize
from skimage.exposure import equalize_adapthist


def load_dicom_series(directory):
    """Carga una serie DICOM desde un directorio y devuelve el volumen y el espaciado."""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(directory)
    
    if not dicom_files:
        raise FileNotFoundError(f"No se encontraron archivos DICOM en {directory}")

    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    
    # Reordenar a (X, Y, Z)
    volume = np.transpose(volume, (2, 1, 0))
    spacing = image.GetSpacing()[::-1]  # (X, Y, Z)

    return volume, spacing


def load_and_preprocess_ct_scan(input_path, target_size=(128, 128, 128)):
    """
    Carga y preprocesa un volumen CT.
    Soporta directorios DICOM, archivos DICOM individuales y archivos ZIP.
    
    Args:
        input_path (str): Ruta a los datos.
        target_size (tuple): Tamaño objetivo del volumen.
    
    Returns:
        volume_preprocessed (np.ndarray): Volumen normalizado y preprocesado.
        original_volume (np.ndarray): Volumen reescalado sin preprocesar.
    """
    temp_dir = None
    try:
        if isinstance(input_path, str) and input_path.endswith('.zip'):
            temp_dir = 'temp_dicom'
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            input_path = temp_dir

        if os.path.isdir(input_path):
            volume, spacing = load_dicom_series(input_path)
        elif input_path.lower().endswith('.dcm'):
            ds = pydicom.dcmread(input_path)
            volume = ds.pixel_array
            spacing = (1.0, 1.0, 1.0)  # Se asume espaciado
        else:
            raise ValueError("Ruta de entrada no soportada: debe ser directorio, .dcm o .zip")

        # Guardar copia reescalada del volumen original
        original_volume = resize(volume, target_size, mode='reflect', anti_aliasing=True)

        # Preprocesamiento completo
        volume_preprocessed = preprocess_volume(volume, spacing, target_size)

        return volume_preprocessed, original_volume

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def preprocess_volume(volume, spacing, target_size):
    """
    Aplica preprocesamiento estándar a un volumen CT:
    1. Ventana pulmonar
    2. Resample a voxel isotrópico
    3. Resize a tamaño uniforme
    4. Ecualización de contraste
    5. Normalización [0, 1]
    """
    volume = apply_lung_window(volume)

    volume = resample_volume(volume, spacing)

    volume = resize(volume, target_size, mode='reflect', anti_aliasing=True)

    # Aplicar CLAHE slice por slice
    for i in range(volume.shape[0]):
        volume[i] = equalize_adapthist(volume[i])

    # Normalizar intensidades a [0, 1]
    min_val, max_val = volume.min(), volume.max()
    if max_val - min_val != 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)

    return volume


def apply_lung_window(volume, level=-600, width=1500):
    """
    Aplica la ventana pulmonar al volumen para mejorar el contraste en estructuras de interés.

    Args:
        level (int): Nivel central de la ventana.
        width (int): Ancho de la ventana.

    Returns:
        volume (np.ndarray): Volumen recortado y normalizado.
    """
    window_min = level - width / 2
    window_max = level + width / 2
    volume = np.clip(volume, window_min, window_max)
    volume = (volume - window_min) / (window_max - window_min)
    return volume


def resample_volume(volume, original_spacing, target_spacing=1.0):
    """
    Re-muestrea el volumen a un espaciado isotrópico (target_spacing).
    
    Args:
        volume (np.ndarray): Volumen original.
        original_spacing (tuple): Espaciado original (X, Y, Z).
        target_spacing (float): Nuevo espaciado isotrópico.
    
    Returns:
        volume (np.ndarray): Volumen re-muestreado.
    """
    resize_factors = [os / target_spacing for os in original_spacing]

    new_shape = tuple(
        max(1, int(round(dim * factor)))
        for dim, factor in zip(volume.shape, resize_factors)
    )

    volume_resampled = resize(
        volume,
        output_shape=new_shape,
        mode='reflect',
        anti_aliasing=True
    )
    return volume_resampled
