# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/nvidia.csv"
PROCESSED_PATH = "data/processed_nvidia.csv"
MODEL_PATH     = "models/xgboost_model.pkl"
SCALER_PATH    = "models/scaler.pkl"
REPORTS_DIR    = "reports"

# Only these 4 features will be used
FEATURES = ["Open", "High", "Low", "Close"]
TARGET   = "Target"

def save_plot(filename):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{filename}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Saved → {path}")
    plt.show()


# ── Phase 1: EDA & Preprocessing ─────────────────────────────────────────────
def run_eda_and_preprocess():
    print("\n--- Phase 1: EDA & Preprocessing ---")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    # EDA: Closing Price Trend
    plt.figure(figsize=(12, 5))
    plt.plot(df['Close'], color='green')
    plt.title("NVIDIA Closing Price Trend")
    save_plot("eda_price_trend")

    # EDA: Correlation Heatmap (only the 4 features)
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    save_plot("eda_correlation")

    # Target: next day's closing price
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    df.to_csv(PROCESSED_PATH)
    print(f"[INFO] Processed data saved → {PROCESSED_PATH}")
    return df


# ── Phase 2: Model Training ───────────────────────────────────────────────────
def train_model(df):
    print("\n--- Phase 2: Model Training ---")

    X = df[FEATURES]   # only 4 columns
    y = df[TARGET]

    # Time-series split — 90% train / 10% test (no shuffling)
    split = int(len(df) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit scaler ONLY on training data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # XGBoost
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=100
    )

    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    print(f"[INFO] Model saved → {MODEL_PATH}")
    print(f"[INFO] Scaler saved → {SCALER_PATH}")

    return model, scaler, X_test_scaled, y_test


# ── Phase 3: Evaluation ───────────────────────────────────────────────────────
def evaluate_model(model, X_test_scaled, y_test):
    print("\n--- Phase 3: Evaluation ---")
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Actual vs Predicted
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Actual",    color='blue',   linewidth=1.5)
    plt.plot(y_pred,        label="Predicted", color='orange', linestyle='--')
    plt.title("Actual vs Predicted Closing Price")
    plt.legend()
    save_plot("eval_actual_vs_pred")

    # Feature Importance
    plt.figure(figsize=(6, 4))
    importances = model.feature_importances_
    sns.barplot(x=importances, y=FEATURES, palette='viridis')
    plt.title("XGBoost Feature Importance")
    save_plot("eval_feature_importance")

    # Residuals
    plt.figure(figsize=(8, 5))
    residuals = y_test.values - y_pred
    sns.histplot(residuals, kde=True, color='steelblue')
    plt.title("Residual Error Distribution")
    save_plot("eval_residuals")

    # Metrics bar
    plt.figure(figsize=(5, 4))
    sns.barplot(x=['RMSE', 'MAE'], y=[rmse, mae])
    plt.title("Error Metrics Summary")
    save_plot("eval_metrics_summary")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_eda_and_preprocess()
    model, scaler, X_test_scaled, y_test = train_model(df)
    evaluate_model(model, X_test_scaled, y_test)
    print("\n✅ Pipeline complete! Check the 'reports/' folder for all plots.")