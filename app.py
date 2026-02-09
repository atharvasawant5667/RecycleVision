import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="RecycleVision", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "garbage_classifier_final.keras",
        compile=False
    )

model = load_model()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_map = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}


st.title("♻️ RecycleVision")
st.subheader("Garbage Image Classification using Deep Learning")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class = class_map[np.argmax(pred)]
    confidence = np.max(pred) * 100


    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {pred_class} ({confidence:.2f}%)")



