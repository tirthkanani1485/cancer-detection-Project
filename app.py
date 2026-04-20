import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load('model/cancer_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Page config
st.set_page_config(page_title="Cancer Detection App", page_icon="🧬", layout="wide")

# ---------- LANDING PAGE ----------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: #38bdf8;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 40px;
    }
    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧬 Cancer Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Breast Cancer Prediction using Machine Learning</div>', unsafe_allow_html=True)

# ---------- INPUT SECTION ----------
st.markdown("### 🔍 Enter Medical Details")

col1, col2, col3 = st.columns(3)

with col1:
    radius = st.number_input("Radius Mean", value=10.0)

with col2:
    texture = st.number_input("Texture Mean", value=10.0)

with col3:
    area = st.number_input("Area Mean", value=500.0)

# ---------- PREDICTION ----------
if st.button("Predict 🔬"):
    input_data = np.array([[radius, texture, area]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.markdown("### 🧾 Result")

    if prediction == 1:
        st.error("⚠️ Cancer Detected")
    else:
        st.success("✅ No Cancer Detected")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Developed by Tirth Kanani 🚀")


from src.predict import predict
result = predict(radius, texture, area)
st.write(result)