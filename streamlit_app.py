# streamlit_app.py

import streamlit as st
from src.pipeline.predict import PredictionPipeline
import tempfile
import subprocess

st.set_page_config(page_title="Brain Tumor Classification", layout="centered")
st.title("Brain Tumor Classification")


uploaded_file = st.file_uploader("Please Upload a Brain MRI", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Picture", use_container_width=True)


    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_image_path = temp_file.name


    pipeline = PredictionPipeline(temp_image_path)
    result = pipeline.predict()

    st.success(f"Model Result: {result[0]['image']}")


if st.button("Re-train Model"):
    with st.spinner("Model is Training ...."):
        process = subprocess.run(["python", "main.py"], capture_output=True, text=True)
        if process.returncode == 0:
            st.success("🎉 Training is Successful")
        else:
            st.error("❌Model Training is Failure")
            st.text(process.stderr)