import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# Load saved model, encoder, and feature columns
@st.cache_resource
def load_resources():
    model = joblib.load("xgb_model.pkl")
    encoder = joblib.load("encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, encoder, feature_columns

model, encoder, feature_columns = load_resources()

# Heartbeat animation on load
with st.spinner('Loading App...'):
    st.markdown("""
        <div style="display:flex; justify-content:center; margin-top:50px;">
            <div class="heartbeat">❤️</div>
        </div>
        <style>
        .heartbeat {
            font-size: 60px;
            animation: beat 1s infinite;
        }

        @keyframes beat {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.3); }
        }
        </style>
    """, unsafe_allow_html=True)
    time.sleep(2)

# Custom dark mode styling
st.markdown("""
    <style>
    body {
        background-color: black !important;
        color: white !important;
    }

    .stApp {
        background-color: black !important;
        color: white !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: white !important;
    }

    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1a1a;
        color: white;
        border-radius: 5px;
    }

    .stButton>button {
        background-color: #d62828;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 0.75em;
        transition: 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #a61b1b;
        transform: scale(1.05);
    }

    .block-container {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("## ❤️ Heart Disease Prediction App")
st.markdown("*Enter the patient's data below to assess the risk of heart disease.*")

# Form layout using 2-column structure
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        chol = st.number_input("Serum Cholesterol (chol)", value=200)
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        slope = st.selectbox("Slope of ST (slope)", [0, 1, 2])

    with col2:
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "👧 Female" if x == 0 else "👦 Male")
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        thalach = st.number_input("Max Heart Rate (thalach)", value=150)
        oldpeak = st.number_input("ST depression (oldpeak)", value=1.0, step=0.1)
        ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    submitted = st.form_submit_button("🔢 Predict")

# Handle prediction
if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    categorical_indices = [2, 6, 10, 11, 12]
    numerical_indices = [0, 1, 3, 4, 5, 7, 8, 9]

    input_categorical = input_data[:, categorical_indices]
    input_numerical = input_data[:, numerical_indices]

    input_encoded = encoder.transform(input_categorical)
    input_df = pd.DataFrame(
        np.concatenate([input_numerical, input_encoded], axis=1),
        columns=feature_columns
    )

    proba = model.predict_proba(input_df)[0][1]
    result = "🔴 Positive for Heart Disease" if proba >= 0.5 else "🟢 No Heart Disease Detected"

    st.markdown("##")
    st.markdown(f"### Prediction: **{result}**")
    st.metric("Probability", f"{proba:.2f}")
