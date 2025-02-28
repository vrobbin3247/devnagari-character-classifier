
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

st.title("Devnagari Character Classifier")

def load_model():
    model = tf.keras.models.load_model("classifier86v3.keras")
    return model



canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=12,
    stroke_color="#000000",
    background_color="#ffffff",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

class_labels = [
   'ज्ञ', 'ट', 'ठ', 'ड',
   'ढ', 'ण', 'त', 'थ',
   'द', 'ध', 'क', 'न', 'प',
   'फ', 'ब', 'भ', 'म', 'य',
   'र', 'ल', 'व', 'ख', 'श',
   'ष', 'स', 'ह', 'क्ष',
   'त्र', 'ज्ञ', 'ग', 'घ', 'ङ',
   'च', 'छ', 'ज', 'ऋ', '0', '१',
   '२', '३', '४', '५', '६', '७', '८', '९'
]

def predict_character():
    if canvas_result.image_data is None:
        st.warning("Please draw a character first!")
        return

    image_data = canvas_result.image_data
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)

    img = cv2.resize(gray_image, (32, 32))
    img = img / 255.0

    img = np.expand_dims(img, axis=(0, -1))

    st.image(img, caption="Processed Image (32x32 Grayscale)", width=100, clamp=True)

    model = load_model()

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    # print(predictions)
    st.success(f"Predicted Character: **{predicted_label}**")

# Button to trigger prediction
st.button("Predict Character", on_click=predict_character)
