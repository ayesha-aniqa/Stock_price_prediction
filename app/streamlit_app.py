import streamlit as st
import pickle
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NVDA Predictor", layout="wide")

# ── CSS: keep original teal/mint look ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #e8f5f3;
}

/* Sidebar-style left column */
section[data-testid="column"]:first-child {
    background-color: #d0ede8;
    border-radius: 0;
    padding: 32px 24px !important;
    min-height: 100vh;
}

/* Input labels */
.stNumberInput label {
    color: #2d6a62 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    margin-bottom: 2px;
}

/* Input boxes */
.stNumberInput input {
    background-color: #ffffff !important;
    border: 1px solid #a8d5ce !important;
    border-radius: 8px !important;
    color: #1a3d38 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
}
.stNumberInput input:focus {
    border-color: #3a9e8f !important;
    box-shadow: 0 0 0 2px rgba(58,158,143,0.2) !important;
}

/* Button */
.stButton > button {
    background-color: #3a9e8f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 10px 28px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: auto !important;
    margin-top: 12px;
    transition: background 0.2s ease;
}
.stButton > button:hover {
    background-color: #2d8070 !important;
}

/* Main right panel */
section[data-testid="column"]:last-child {
    background-color: #f0faf8;
    padding: 48px 48px !important;
}

/* Title */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1a3d38;
    margin-bottom: 4px;
}
.main-subtitle {
    font-size: 0.95rem;
    color: #4a7a74;
    margin-bottom: 36px;
}

/* Result box */
.result-box {
    background-color: #d6f0ec;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 20px;
    font-size: 1rem;
    color: #1a3d38;
    font-weight: 500;
}
.result-price {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a3d38;
}

/* Section label */
.section-label {
    font-size: 1rem;
    font-weight: 600;
    color: #1a3d38;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

# Left column — inputs
with col1:
    st.markdown('<p class="section-label">Input Market Values</p>', unsafe_allow_html=True)

    open_p  = st.number_input("Open Price",  min_value=0.0, value=100.0, step=0.5, format="%.2f")
    high_p  = st.number_input("High Price",  min_value=0.0, value=105.0, step=0.5, format="%.2f")
    low_p   = st.number_input("Low Price",   min_value=0.0, value=98.0,  step=0.5, format="%.2f")
    close_p = st.number_input("Close Price", min_value=0.0, value=102.0, step=0.5, format="%.2f")

    predict_btn = st.button("Predict Closing Price")

# Right column — results
with col2:

    st.markdown('<h1 class="main-title">📈 NVDA Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<h6 class="main-subtitle">Enter the market data below to predict the <strong>Closing Price</strong>.</h6>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not found. Please run pipeline.py first.")
    elif predict_btn:
        raw    = np.array([[open_p, high_p, low_p, close_p]])
        scaled = scaler.transform(raw)
        pred   = model.predict(scaled)[0]
        delta  = pred - close_p
        sign   = "+" if delta >= 0 else ""

        st.markdown(f"""
        <div class="result-box">
            The predicted Closing Price is: <span class="result-price">${pred:,.2f}</span>
            &nbsp;&nbsp;
            <span style="font-size:0.9rem; color:{'#1a6b50' if delta >= 0 else '#a33030'}">
                ({sign}{delta:.2f} from current close)
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-box" style="opacity:0.5;">
            Fill in the market values on the left and click <strong>Predict Closing Price</strong>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#7aada6; font-size:0.78rem;">
        For educational purposes only. Not financial advice.
    </p>
    """, unsafe_allow_html=True)