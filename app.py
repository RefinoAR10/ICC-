import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model files
model = joblib.load("cricket_model.pkl","r")
scaler = joblib.load("scaler.pkl","r")
columns = joblib.load("columns.pkl","r")

# App Title
st.title("🏏 Cricket Player Performance Prediction")

st.write("Enter Player Statistics")

# User Inputs
Mat = st.slider("Matches", min_value=0, max_value=500, value=50)

Inns = st.slider("Innings", min_value=0, max_value=500, value=40)

NO = st.slider("Not Outs", min_value=0, max_value=100, value=5)

Runs = st.slider("Runs", min_value=0, max_value=20000, value=500)

HS = st.slider("Highest Score", min_value=0, max_value=500, value=100)

Ave = st.slider("Average", min_value=0.0, max_value=100.0, value=35.0)

BF = st.slider("Balls Faced", min_value=0, max_value=100000, value=1000)

SR = st.slider("Strike Rate", min_value=0.0, max_value=300.0, value=80.0)

Hundreds = st.slider("100s", min_value=0, max_value=100, value=2)

Fifties = st.slider("50s", min_value=0, max_value=100, value=5)

Ducks = st.slider("Zeros", min_value=0, max_value=100, value=1)

start = st.slider("Start Year", min_value=1900, max_value=2025, value=2010)

end = st.slider("End Year", min_value=1900, max_value=2025, value=2020)

# Experience
exp = end - start

# Prediction Button
if st.button("Predict Performance"):

    # Create DataFrame
    input_data = pd.DataFrame([[
        Mat, Inns, NO, Runs, HS,
        Ave, BF, SR, Hundreds,
        Fifties, Ducks,
        start, end, exp
    ]], columns=columns)

    # Scale Data
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)

    # Output
    if prediction[0] == 2:
        st.success("⭐ Best Performance Player")

    elif prediction[0] == 1:
        st.warning("⚡ Average Performance Player")

    else:
        st.error("❌ Low Performance Player")