import pandas as pd
import cv2
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import ultralytics
from ultralytics import YOLO
import numpy as np

st.image("logo-upch-ing-des.webp", use_column_width=True)

# Importando MODELO
model = YOLO('best.pt')
model2 = YOLO('last.pt')

tab1, tab2 = st.tabs(["Haz un  dibujito ✍️", "Sube una imágen ⬆️"])

with tab1:
    import streamlit as st

    st.header("Haz un dibujito")

    # Especifica los parametros canvas de la aplicacion canvas
    drawing_mode = st.selectbox(
        "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
    )

    stroke_width = st.slider("Ancho del trazo: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.slider("Tamaño del punto: ", 1, 25, 3)
    stroke_color = st.color_picker("Color: ")
    # FFFDFD - Color blanco del fondo on default, necesario
    bg_color = st.color_picker("Background color hex: ", "#FFFDFD")

    realtime_update = st.checkbox("Actualizar en tiempo real", True)

    modelos_disponibles = {'Modelo 1 (best.pt)': model, 'Modelo 2 (last.pt)': model2}
    option = st.selectbox("Escoja modelo: ", list(modelos_disponibles.keys()), key="select_model_tab1")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        # box draw size
        height=640,  # Ajustado alto
        width=640,  # Ajustado ancho
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas"
    )

    # Boton predecir:
   
    # Boton predecir:
    if st.button("Predecir dibujo"):
        if canvas_result.image_data is not None:
            # Convertir la imagen a formato RGBA
            image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

            # Convertir la imagen a formato BGR
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convertir la imagen a formato RGB antes de pasarla al modelo
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Obtener el modelo seleccionado
            selected_model = modelos_disponibles[option]

            # Config del modelo
            config = {
                "conf": 0.2,
                "imgsz": 640,
            }

            # Realizar predicción en el dibujo con configuración
            results = selected_model(image_rgb, **config)

            # Obtener la imagen anotada
            annotated_image = results[0].plot()

            # Mostrar la imagen resultante
            st.image(annotated_image, channels="RGB")
        else:
            st.write("No hay imagen")
with tab2:
    import streamlit as st

    st.header("Sube una imágen")
    source_img = st.file_uploader(
        "Escoge una imágen...", type=("jpg", "jpeg", "png", "bmp", "webp"))

    modelos_disponibles = {'Modelo 1 (best.pt)': model, 'Modelo 2 (last.pt)': model2}
    option = st.selectbox("Escoja modelo: ", list(modelos_disponibles.keys()), key="select_model_tab2")

    if st.button("Predecir imagen"):
        if source_img is not None:
            st.image(source_img, channels="BGR")  # Mostrar la imagen original

            # Leer la imagen subida con OpenCV sin modificar el color
            file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)  # Cargar sin modificar los canales

            # Obtener el modelo seleccionado
            selected_model = modelos_disponibles[option]

            # Config del modelo
            config = {
                "conf": 0.2,
                "imgsz": 640,
            }

            # Realizar predicción en la imagen con configuración
            results = selected_model(image, **config)

            # Crear una nueva imagen anotada a partir de los datos de predicción
            annotated_image = results[0].plot()

            # Mostrar la imagen anotada con los colores originales
            st.image(annotated_image, channels="BGR")
        else:
            st.write("No hay imagen")