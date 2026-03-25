import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# -------------------------------
# 📥 Download model from Google Drive
# -------------------------------
MODEL_PATH = "best_alzheimer_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = "https://drive.google.com/uc?id=1rijxIT4FveeKIzwgW8CGvl0lvkHXiy2r"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# 🧠 Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# -------------------------------
# 🎯 Class names
# -------------------------------
classes = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

# -------------------------------
# 🖼️ Image preprocessing
# -------------------------------
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# 🎨 UI
# -------------------------------
st.set_page_config(page_title="Alzheimer Detection", layout="centered")

st.title("🧠 Alzheimer’s Disease Detection")
st.write("Upload an MRI image to predict dementia stage")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# 🔍 Prediction
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        img = preprocess_image(image)
        prediction = model.predict(img)
        probs = prediction[0]

        idx = np.argmax(probs)
        result = classes[idx]
        confidence = probs[idx] * 100

    # -------------------------------
    # 📊 Output
    # -------------------------------
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")

    # -------------------------------
    # 📈 Probability display
    # -------------------------------
    st.subheader("Prediction Probabilities")

    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probs[i]*100:.2f}%")

    # -------------------------------
    # ⚠️ Recommendation
    # -------------------------------
    st.subheader("Recommendation")

    if result == "NonDemented":
        st.success("No signs of dementia detected.")
    elif result == "VeryMildDemented":
        st.warning("Very mild cognitive impairment.")
    elif result == "MildDemented":
        st.warning("Early stage dementia detected.")
    else:
        st.error("Moderate dementia detected. Consult doctor immediately.")

# -------------------------------
# ℹ️ Footer
# -------------------------------
st.markdown("---")
st.caption("⚠️ This app is for educational purposes only.")
