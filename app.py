import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np

st.set_page_config(page_title="YOLOv8 Inference App", layout="centered")
st.title("🔍 YOLOv8 Image Detector")

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

image_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model(image)
        result_img = results[0].plot()  # This is a NumPy array

    # Show the result
    st.image(result_img, caption="🧠 Detection Result", use_column_width=True)

    # Save as image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        Image.fromarray(result_img).save(tmp.name)  # Convert NumPy to PIL before saving
        st.download_button("📥 Download Output", data=open(tmp.name, "rb").read(), file_name="detected.jpg")

else:
    st.info("Please upload an image to begin detection.")
