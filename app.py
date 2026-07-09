import streamlit as st
# Hide the default Streamlit menu, header, and footer
st.set_page_config(page_title="Loan Default Predictor", layout="centered")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import pandas as pd
import joblib

# Load model and preprocessing artifacts
model = joblib.load("tree_loan.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Default Loan Prediction by Rajan")
st.markdown("Provide the following Details:")

# Input fields
Age = st.slider("Age", 18, 100, 40)
Income = st.slider("Income", 30000, 150000, 50000)
Credit_score = st.slider("Credit_score", 500, 800, 650)
Debt = st.slider("Debt", 1000, 50000, 15000)
employment_status = st.selectbox("Employment Status", ['Self-Employed','Unemployed','Employed'])

# Prediction only runs if button is clicked
if st.button("Predict"):

    # Create input dictionary
    raw_input = {
        "Age": Age,
        "Income": Income,
        "Credit_Score": Credit_score,
        "Existing_Debt": Debt,
        f"Employment_status_{employment_status}": 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0    

    # Ensure column order
    input_df = input_df[expected_columns]

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == "Yes":   # Assuming "Yes" = Default
        st.error("⚠️ Failed to repay")
    else:
        st.success("✅ lone repaid successfully.")
