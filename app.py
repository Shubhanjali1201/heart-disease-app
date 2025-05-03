# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model, encoder, and feature columns
@st.cache_resource
def load_resources():
    model = joblib.load("xgb_model.pkl")
    encoder = joblib.load("encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, encoder, feature_columns

model, encoder, feature_columns = load_resources()

# Streamlit UI
st.title("💓 Heart Disease Prediction App")
st.write("Enter the patient's data below:")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
    chol = st.number_input("Serum Cholesterol (chol)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", value=1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
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
    result = "🟥 Positive for Heart Disease" if proba >= 0.5 else "🟩 No Heart Disease Detected"

    st.subheader(f"Prediction: {result}")
    st.write(f"Prediction Probability: {proba:.2f}")
