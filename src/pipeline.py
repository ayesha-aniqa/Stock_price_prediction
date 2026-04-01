# -*- coding: utf-8 -*-

# Import Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("C:\\Users\\Shaitan Biliii\\Desktop\\Stock_price_prediction\\data\\processed_nvidia.csv")

# 1. basic Exploration
print(df.info())
print(df.describe())

DATA_PATH = "C:\\Users\\Shaitan Biliii\\Desktop\\Stock_price_prediction\\data\\nvidia.csv"
PROCESSED_PATH = "C:\\Users\\Shaitan Biliii\\Desktop\\Stock_price_prediction\\data\\processed_nvidia.csv"
MODEL_PATH = "C:\\Users\\Shaitan Biliii\\Desktop\\Stock_price_prediction\\models\\xgboost_model.pkl"
SCALER_PATH = "C:\\Users\\Shaitan Biliii\\Desktop\\Stock_price_prediction\\models\\scaler.pkl"
REPORTS_DIR = "C:\\Users\\Shaitan Biliii\\Desktop\\Stock_price_prediction\\reports"

def save_plot(filename):
    """Automatically saves the current plot to the reports folder."""
    path = os.path.join(REPORTS_DIR, f"{filename}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Graph saved to {path}")
    plt.show()

"""# EDA"""

# ==========================================
# 1. EDA & PREPROCESSING SECTION
# ==========================================
def run_eda_and_preprocess():
    print("\n--- Phase 1: EDA & Preprocessing ---")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    # --- EDA GRAPHS ---
    # 1. Price Trend
    plt.figure(figsize=(12,5))
    plt.plot(df['Close'], color='green')
    plt.title("Nvidia Closing Price Trend")
    save_plot("eda_price_trend")

    # 2. Correlation Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    save_plot("eda_correlation")

    # --- FEATURE ENGINEERING ---
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)

    # Target: We want to predict the next day's price
    df["Target"] = df["Close"].shift(-1)

    # Handle missing values from rolling windows
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # 3. Moving Average Visualization
    plt.figure(figsize=(12,5))
    plt.plot(df['Close'], label='Price', alpha=0.5)
    plt.plot(df['MA50'], label='50-Day MA', color='red')
    plt.title("Price vs Moving Average")
    plt.legend()
    save_plot("eda_moving_average")

    df.to_csv(PROCESSED_PATH)
    return df

"""# Model Training (XGBoost)"""

# MODEL TRAINING
def train_model(df):
    print("\n--- Phase 2: Model Training ---")

    # Define Features and Target
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Time-Series Split (90% Train / 10% Test)
    split = int(len(df) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scaling - Crucial: Fit ONLY on train data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost Regressor
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

    # Save Pickle Files
    with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)

    return model, scaler, X_test_scaled, y_test, X.columns

"""# Evaluation and Testing

"""

# 3. EVALUATION SECTION
def evaluate_model(model, scaler, X_test_scaled, y_test, feature_names):
    print("\n--- Phase 3: Evaluation ---")
    y_pred = model.predict(X_test_scaled)

    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"METRICS >> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    # --- EVALUATION GRAPHS ---

    # 1. Actual vs Predicted
    plt.figure(figsize=(12,6))
    plt.plot(y_test.values, label="Actual Price", color='blue', linewidth=1.5)
    plt.plot(y_pred, label="XGBoost Prediction", color='orange', linestyle='--')
    plt.title("Final Model: Actual vs Predicted Prices")
    plt.legend()
    save_plot("eval_actual_vs_pred")

    # 2. Feature Importance
    plt.figure(figsize=(10,6))
    importances = model.feature_importances_
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title("XGBoost Feature Importance")
    save_plot("eval_feature_importance")

    # 3. Residual Error
    plt.figure(figsize=(8,5))
    residuals = y_test.values - y_pred
    sns.histplot(residuals, kde=True, color='purple')
    plt.title("Residual Error Distribution")
    save_plot("eval_residuals")

    # 4. Metrics Bar Chart
    plt.figure(figsize=(6,4))
    metric_names = ['RMSE', 'MAE']
    metric_values = [rmse, mae]
    sns.barplot(x=metric_names, y=metric_values)
    plt.title("Error Metrics Summary")
    save_plot("eval_metrics_summary")

# MAIN EXECUTION
if __name__ == "__main__":
    # Run Preprocessing
    processed_df = run_eda_and_preprocess()

    # Run Training
    model, scaler, X_test_s, y_test, feat_names = train_model(processed_df)

    # Run Evaluation
    evaluate_model(model, scaler, X_test_s, y_test, feat_names)

    print("\n✅ Pipeline complete! Check the 'reports' folder for all visual artifacts.")

# 2. Check for missing values
print("Missing values:\n", df.isnull().sum())

# 3. Visualize the Closing Price over time
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'], label='Nvidia Close Price')
plt.title('Nvidia Historical Close Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()