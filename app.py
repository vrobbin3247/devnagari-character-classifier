
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

st.title("Devanagari Character Classifier")
def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns([3, 1, 3, 1, 3])
st.markdown(
    """
    <style>
    [data-testid="stHorizontalBlock"] {
            align-items: center;
    }

    """,
    unsafe_allow_html=True,
)
with col1:
    canvas_result = st_canvas(
        stroke_width=30,
        stroke_color="#ffffff",
        background_color="#000000",
        height=320,
        width=320,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True
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
    model = load_model()
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    return predicted_label


def preprocess():
    if canvas_result.image_data is None:
        st.warning("Please draw a character first!")
        return

    image_data = canvas_result.image_data
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)

    img = cv2.resize(gray_image, (32, 32))
    img = img / 255.0

    img = np.expand_dims(img, axis=(0, -1))

    # st.image(img, caption="Processed Image (32x32 Grayscale) 10x scaled", width=320, clamp=True)

    return img

if "preprocessed_image" not in st.session_state:
    st.session_state.preprocessed_image = None
if "predicted_image" not in st.session_state:
    st.session_state.predicted_image = None
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None

# Preprocess Button
with col2:
    if st.button("process", key="preprocess", type='primary'):
        st.session_state.preprocessed_image = preprocess()

with col3:
    if st.session_state.preprocessed_image is not None:
        st.image(st.session_state.preprocessed_image, caption="Preprocessed Image (32x32 Grayscale) 10x scaled",
                 width=320)

with col4:
    # Predict Button
    if st.button("predict", key="predict", type='primary') and st.session_state.preprocessed_image is not None:
        predicted_label = predict_character(st.session_state.preprocessed_image)
        st.session_state.predicted_label = predicted_label
        st.session_state.predicted_image = f"characters/{predicted_label}.png"

with col5:
    image_placeholder = st.empty()

    if st.session_state.predicted_image is not None and st.session_state.predicted_label is not None:
        image_placeholder.image("loading.gif", caption="Analyzing...", )

        time.sleep(2)

        image_placeholder.image(
            st.session_state.predicted_image,
            caption=f"Predicted: {st.session_state.predicted_label}"
        )
