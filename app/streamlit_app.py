import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# 1. Page Setup
st.set_page_config(page_title="NVDA Predictor", layout="wide")

# Custom CSS for Pastel Colors
st.markdown("""
    <style>
    .stApp { background-color: #F0F4F8; }
    .stButton>button { 
        background-color: #B2DFDB; 
        color: #455A64; 
        border-radius: 20px; 
        width: 100%;
    }
    h1, h2, h3 { color: #546E7A; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 NVIDIA Stock Price Predictor")

# 2. Load the Model and Scaler
@st.cache_resource
def load_assets():
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Model or Scaler not found! Please run src/pipeline.py first.")
    st.stop()

# 3. Create the UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Market Data")
    open_p = st.number_input("Open Price", value=120.0)
    high_p = st.number_input("High Price", value=125.0)
    low_p  = st.number_input("Low Price", value=118.0)
    curr_p = st.number_input("Current Close", value=122.0)
    vol    = st.number_input("Volume", value=50000000)
    
    # Simple Technical Inputs (To match your Pipeline features)
    ma50 = st.number_input("50-Day Moving Average", value=115.0)
    lag1 = st.number_input("Yesterday's Close", value=121.0)

# 4. Prediction Logic
with col2:
    st.header("Prediction Result")
    if st.button("Predict Tomorrow's Price"):
        # Note: The order here MUST match exactly the order of features in your X_train
        # From your pipeline: ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'Return', 'Volatility', 'Lag1', 'Lag2', 'Lag3']
        # For simplicity, we will fill missing technicals with current values
        
        raw_features = np.array([[open_p, high_p, low_p, curr_p, vol, curr_p, ma50, 0.01, 0.02, lag1, lag1, lag1]])
        
        # 1. Scale the features
        scaled_features = scaler.transform(raw_features)
        
        # 2. Predict
        prediction = model.predict(scaled_features)
        
        # 3. Show Result
        st.metric(label="Predicted Price", value=f"${prediction[0]:.2f}", delta=f"{prediction[0] - curr_p:.2f}")
        
