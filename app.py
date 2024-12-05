import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# Configuración de la página
st.set_page_config(page_title="Clasificador de Imágenes y Videos", layout="centered")

# Cargar el modelo
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

model_path = "models/image_net_v3_best_model.keras"
model = load_trained_model(model_path)

# Diccionario de clases
class_names = {0: "Nonviolence", 1: "Violence"}

# Procesamiento de imágenes
def preprocess_image(image):
    image_size = 256
    image = cv2.resize(image, (image_size, image_size))
    image = img_to_array(image)  # Convertir a matriz numpy
    image = np.expand_dims(image, axis=0)  # Agregar dimensión para lotes
    image = image / 255.0  # Normalización

    return image

# Procesamiento de videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocesar el cuadro
        processed_frame = preprocess_image(frame)
        prediction = model.predict(processed_frame)
        predicted_class = int(prediction[0] > 0.5)
        confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

        frame_results.append((frame, predicted_class, confidence))
    cap.release()

    return frame_results

# Interfaz de usuario
st.title("Clasificador de Imágenes y Videos con contenido violento mediante la arquitectura ResNet50")
st.write("Sube una imagen o un video para clasificarlo.")

# Opción para seleccionar tipo de archivo
option = st.selectbox("Selecciona el tipo de archivo", ["Imagen", "Video"])

if option == "Imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Preprocesar y predecir
        st.write("Procesando la imagen...")
        image = np.array(image)  # Convertir a formato numpy
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)

        # Interpretar predicción
        predicted_class = int(prediction[0] > 0.5)
        confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

        # Mostrar resultado
        st.write(f"**Clase Predicha:** {class_names[predicted_class]}")
        st.write(f"**Confianza:** {confidence:.2%}")

elif option == "Video":
    uploaded_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        st.video(uploaded_file)
        # Guardar el video temporalmente
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())



        # Procesar el video
        st.write("Procesando el video...")
        results = process_video(tfile.name)

        predictions = {}

        # Mostrar resultados cuadro por cuadro
        for frame, predicted_class, confidence in results:

            predictions[predicted_class] = confidence


        violence = [
            predicted_class for predicted_class in predictions.keys() if predicted_class == 1
        ]

        percentage_violence = len(violence) / len(predictions.keys())

        predicted_class = int(percentage_violence > 0.5)
        confidence = np.average(list(predictions.values()))

        # Mostrar resultado
        st.write(f"**Clase Predicha:** {class_names[predicted_class]}")
        st.write(f"**Confianza:** {confidence:.2%}")


        # Eliminar el archivo temporal
        os.unlink(tfile.name)

# Footer
st.write("---")
st.write(
    "Desarrollado por: Juan David Sepúlveda (j.sepulveda@uniandes.edu.co) Emil Rueda (vr.rueda@uniandes.edu.co)"
)
