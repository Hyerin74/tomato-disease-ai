import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 모델 파일 다운로드
model_path = "best_model.h5"
if not os.path.exists(model_path):
    file_id = "1VDarRTNhj3nlc8haTf9b0XdnHFv4kRBV"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# 모델 로드
model = load_model(model_path)

# Streamlit 앱 인터페이스
st.title("토마토 질병 진단 AI")
uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="업로드된 이미지", use_column_width=True)

    # 전처리
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        st.warning(f"🚨 질병 가능성 (확률: {prediction:.2f})")
    else:
        st.success(f"✅ 정상 (확률: {1 - prediction:.2f})")