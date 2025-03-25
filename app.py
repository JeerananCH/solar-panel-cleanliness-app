import streamlit as st
st.set_page_config(page_title="Solar Panel Cleanliness Classifier", layout="centered")  # ✅ ต้องอยู่ก่อนทุกคำสั่ง

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

@st.cache_resource
def load_best_model():
    return load_model("solar_cleanliness_model.h5")

model = load_best_model()

st.title("Solar Panel Cleanliness Classifier")
st.markdown("อัปโหลดภาพแผงโซลาร์เซลล์ชนิดโพลีคริสตัลไลน์ แล้วจะบอกว่าแผงนี้ **Clean** หรือ **Dirty**")

uploaded_file = st.file_uploader("📤 อัปโหลดภาพ .jpg หรือ .png", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="📸 ตัวอย่างภาพ", use_container_width=True)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "🧼 Clean" if pred < 0.5 else "🧹 Dirty"
    st.markdown(f"### 🔍 Prediction: **{label}** ({pred:.2f})")
