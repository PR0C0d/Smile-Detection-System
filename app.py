import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.markdown(
    """
    <style>
    .stApp {
        background-color: #C65100;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("😊😐 Smile Detector")

# Load model
model = tf.keras.models.load_model("smile_model.keras")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Preprocessing (MUST match training: 160x160 + normalization)
    image = image.resize((160,160))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0]

    # Debug (helps understanding model behavior)
    st.write("Raw score:", score)

    # Improved prediction logic (reduces wrong confident outputs)
    if score > 0.7:
        st.write("Prediction: Smiling 😊")
        confidence = score

    elif score < 0.3:
        st.write("Prediction: Not Smiling 😐")
        confidence = 1 - score

    else:
        st.write("Prediction: Not sure 🤔")
        confidence = max(score, 1 - score)

    # Confidence display
    st.write("Confidence:", round(confidence * 100, 2), "%")
    st.progress(int(confidence * 100))
