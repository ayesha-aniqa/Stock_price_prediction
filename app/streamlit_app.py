import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# 1. Page Setup
st.set_page_config(page_title="NVDA Predictor", layout="centered")

# Custom CSS for Pastel Colors
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F4F8; /* Cold White-Blue */
    }
    .stButton>button {
        background-color: #B2DFDB; /* Pastel Mint */
        color: #455A64;
        border-radius: 20px;
        border: none;
    }
    h1, h2, h3, p {
        color: #546E7A; /* Cold Slate Gray */
    }
    .stNumberInput div div input {
        background-color: #E1F5FE; /* Ice Blue */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 NVDA Stock Price Predictor")
st.write("Enter the market data below to predict the **Closing Price**.")

# 2. Load the Model
@st.cache_resource
def load_model():
    with open("models/trained_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# 3. Create the UI Inputs
st.sidebar.header("Input Market Values")
open_price = st.sidebar.number_input("Open Price", value=100.0)
high_price = st.sidebar.number_input("High Price", value=105.0)
low_price  = st.sidebar.number_input("Low Price", value=98.0)
volume     = st.sidebar.number_input("Volume", value=50000000)

# 4. Prediction Logic
if st.button("Predict Closing Price"):
    # Arrange inputs exactly like the training data
    features = np.array([[open_price, high_price, low_price, volume]])
    
    prediction = model.predict(features)
    
    # 5. Show Result
    st.success(f"The predicted Closing Price is: **${prediction[0]:.2f}**")
