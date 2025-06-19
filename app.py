import streamlit as st
import joblib
import numpy as np
import gdown
import os

# === Page Config ===
st.set_page_config(page_title="Olympics Medal Region Predictor", layout="centered")

# === Cache and Load Model + Encoders ===
@st.cache_resource
def load_model_and_encoders():
    model_file = "model.pkl"
    model_file_id = "1c1YRNoqJgXxlIfjkPfy5dqenJHk6Sllq"  # Google Drive ID

    if not os.path.exists(model_file):
        gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_file, quiet=False)

    model = joblib.load("model.pkl")
    encoders = joblib.load("label_encoders.pkl")  # Must be in your repo
    return model, encoders

# === Load once using cache ===
model, encoders = load_model_and_encoders()

# === Optional: Blue button style ===
st.markdown("""
    <style>
        .stButton > button {
            background-color: #0066cc;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 16px;
            border: none;
        }

        .stButton > button:hover {
            background-color: #004c99;
        }
    </style>
""", unsafe_allow_html=True)

# === UI ===
st.title("🏅 Olympics Medal Region Predictor")
st.write("Use athlete information to predict which region is most likely to win a medal.")

with st.form("predict_form"):
    st.subheader("🔢 Input Athlete Details")
    sport = st.selectbox("🏃 Sport", encoders["sport"].classes_)
    sex = st.selectbox("⚥ Sex", encoders["sex"].classes_)
    age = st.slider("🎂 Age", 10, 60, 25)
    height = st.slider("📏 Height (cm)", 120, 230, 175)
    weight = st.slider("⚖️ Weight (kg)", 30, 150, 70)
    medal = st.selectbox("🥇 Medal", encoders["medal"].classes_)
    submit = st.form_submit_button("🔍 Predict Region")

if submit:
    try:
        sport_encoded = encoders["sport"].transform([sport])[0]
        sex_encoded = encoders["sex"].transform([sex])[0]
        medal_encoded = encoders["medal"].transform([medal])[0]

        input_data = np.array([[sport_encoded, sex_encoded, age, height, weight, medal_encoded]])
        prediction = model.predict(input_data)[0]
        region_pred = encoders["region"].inverse_transform([prediction])[0]

        st.success(f"🌍 Predicted Region: **{region_pred}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
