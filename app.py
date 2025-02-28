
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

st.markdown(
    """
<style>
button {
    height: auto;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    width: 320px !important;
    content: center !important;
}
</style>
""",
    unsafe_allow_html=True,
)
st.title("Devnagari Character Classifier")

def load_model():
    model = tf.keras.models.load_model("classifier86v3.keras")
    return model



canvas_result = st_canvas(
    stroke_width=30,
    stroke_color="#ffffff",
    background_color="#000000",
    height=320,
    width=320,
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


def predict_character(img):
    # Make prediction
    model = load_model()
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    return predicted_label


def preprocess():
    if canvas_result.image_data is None:
        st.warning("Please draw a character first!")
        return

    # Convert RGBA to Grayscale
    image_data = canvas_result.image_data
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)

    # Resize and normalize
    img = cv2.resize(gray_image, (32, 32))

    # Normalize (Scale pixel values between 0 and 1)
    img = img / 255.0

    # Reshape for model input (batch_size=1, height=32, width=32, channels=1)
    img = np.expand_dims(img, axis=(0, -1))

    # Show the preprocessed image
    # st.image(img, caption="Processed Image (32x32 Grayscale) 10x scaled", width=320, clamp=True)

    return img


# Ensure session state variables exist
if "preprocessed_image" not in st.session_state:
    st.session_state.preprocessed_image = None
if "predicted_image" not in st.session_state:
    st.session_state.predicted_image = None
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None

# Preprocess Button
if st.button("Preprocess", key="preprocess", type='primary'):
    st.session_state.preprocessed_image = preprocess()  # Your preprocessing function

# Display the preprocessed image (if available)
if st.session_state.preprocessed_image is not None:
    st.image(st.session_state.preprocessed_image, caption="Preprocessed Image (32x32 Grayscale) 10x scaled", width=320)

# Predict Button
if st.button("Predict", key="predict", type='primary',
             use_container_width=True) and st.session_state.preprocessed_image is not None:
    predicted_label = predict_character(st.session_state.preprocessed_image)  # Your prediction function
    st.session_state.predicted_label = predicted_label
    st.session_state.predicted_image = f"characters/{predicted_label}.png"

# Display the predicted character image (if available)
if st.session_state.predicted_image is not None and st.session_state.predicted_label is not None:
    st.image(st.session_state.predicted_image, caption=f"Predicted: {st.session_state.predicted_label}")
