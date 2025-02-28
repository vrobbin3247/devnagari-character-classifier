
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

st.title("Devnagari Character Classifier")

def load_model():
    model = tf.keras.models.load_model("classifier86.keras")
    return model



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=20,
    stroke_color="#000000",
    background_color="#ffffff",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# # Display the drawn image
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

class_labels = [
    'character_10_yna', 'character_11_taamatar', 'character_12_thaa', 'character_13_daa',
    'character_14_dhaa', 'character_15_adna', 'character_16_tabala', 'character_17_tha',
    'character_18_da', 'character_19_dha', 'character_1_ka', 'character_20_na', 'character_21_pa',
    'character_22_pha', 'character_23_ba', 'character_24_bha', 'character_25_ma', 'character_26_yaw',
    'character_27_ra', 'character_28_la', 'character_29_waw', 'character_2_kha', 'character_30_motosaw',
    'character_31_petchiryakha', 'character_32_patalosaw', 'character_33_ha', 'character_34_chhya',
    'character_35_tra', 'character_36_gya', 'character_3_ga', 'character_4_gha', 'character_5_kna',
    'character_6_cha', 'character_7_chha', 'character_8_ja', 'character_9_jha', 'digit_0', 'digit_1',
    'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9'
]

def predict_character():
    if canvas_result.image_data is None:
        st.warning("Please draw a character first!")
        return

    # Convert RGBA to Grayscale
    image_data = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Resize to (32, 32)
    img = cv2.resize(image_data, (32, 32))

    # Normalize and reshape
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    model = load_model()
    # Predict the class
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    st.write(f"Predicted Character: **{predicted_label}**")

# Button to trigger prediction
st.button("Predict Character", on_click=predict_character)
