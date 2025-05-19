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

# Option to choose input method
input_method = st.radio("Choose Image Input Method:", ("📤 Upload", "📷 Camera"))

if input_method == "📤 Upload":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)

elif input_method == "📷 Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)

# Proceed if image is available
if ('image' in locals()):
    st.image(image, caption="🖼️ Input Image", use_column_width=True)

    with st.spinner("🔍 Detecting objects..."):
        results = model(image)
        result_img = results[0].plot()

    st.image(result_img, caption="🧠 Detection Result", use_column_width=True)

    # Save detection result
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        Image.fromarray(result_img).save(tmp.name)
        st.download_button("📥 Download Output", data=open(tmp.name, "rb").read(), file_name="detected.jpg")

else:
    st.info("Please provide an image using Upload or Camera.")
