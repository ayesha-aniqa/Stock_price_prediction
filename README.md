📈 Stock Price Prediction using XGBoost
Table of Contents

About The Project

Built With

Getting Started

Dependencies

Alternative: Export Your Environment

Installation

Usage

Roadmap

Contributing

License

Authors

Acknowledgements

About The Project

This project presents a complete Machine Learning pipeline for Stock Price Prediction using historical stock market data. The model is trained to predict future stock prices based on engineered features derived from historical trends.

The project follows a structured ML workflow:

Data Preprocessing using Pandas

Data cleaning

Date indexing

Feature selection

Train-test split (Time-series aware)

Feature scaling using MinMaxScaler

Model Training using XGBoost

XGBRegressor implementation

Hyperparameter configuration

Model fitting on training data

Model Testing & Evaluation

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R² Score

Model Visualization using Matplotlib & Seaborn

Actual vs Predicted Stock Prices

Residual Error Distribution

Feature Importance Plot

Scatter Plot Analysis

The dataset used consists of processed historical stock data (processed_nvidia.csv) containing technical and price-based features for prediction.

This project demonstrates:

Time-series handling

Feature scaling

Gradient boosting regression

Performance evaluation

Data visualization

Model persistence using Pickle

Built With

Python 3.10+

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib

Seaborn

Pickle

Getting Started

To recreate this project locally, follow the steps below.

The repository structure:

Stock_Price_Prediction/
│
├── data/
│   └── processed_nvidia.csv
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_testing.ipynb
│   ├── Model_Visualization.ipynb
│   ├── scaler.pkl
│   └── xGboost_model.pkl
│
├── models/
├── requirements.txt
└── README.md
Dependencies

Below are the required libraries with recommended versions:

pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
xgboost >= 1.7.0
matplotlib >= 3.6.0
seaborn >= 0.12.0

You can install them using:

pip install -r requirements.txt

Or individually:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn
Alternative: Export Your Environment

To export your current conda environment:

conda env export > requirements.yml

Another user can recreate it using:

conda env create -f requirements.yml
Installation

Clone the repository:

git clone https://github.com/your_username/Stock_Price_Prediction.git

Navigate to the project directory:

cd Stock_Price_Prediction

Install dependencies:

pip install -r requirements.txt

Launch Jupyter Notebook:

jupyter notebook
Usage

Follow the notebooks in order:

1️⃣ Data Preprocessing

Run:

notebooks/preprocessing.ipynb

This notebook:

Loads dataset

Cleans and prepares features

Applies scaling

Saves scaler (scaler.pkl)

2️⃣ Model Training

Run:

notebooks/model_training.ipynb

This notebook:

Trains XGBoost Regressor

Fits model on training data

Saves trained model (xGboost_model.pkl)

3️⃣ Model Testing

Run:

notebooks/model_testing.ipynb

This notebook:

Loads saved model

Evaluates on test data

Prints MSE, MAE, and R² score

4️⃣ Model Visualization

Run:

notebooks/Model_Visualization.ipynb

This notebook generates:

📊 Actual vs Predicted Stock Prices

📉 Residual Error Distribution

🔍 Feature Importance Graph

📈 Scatter Plot of Predictions

Example outputs are saved as:

Actual Vs Predicted Stock Price.png

Model Performance Metrics.png

Feature Importance XGBoost.png

Residual Error Distribution.png

Model Performance

The XGBoost model demonstrates strong predictive performance with:

Low Mean Squared Error

Low Mean Absolute Error

High R² Score (close to 1)

The visualization results show a strong alignment between actual and predicted values, indicating good generalization capability.

Roadmap

Future improvements:

Hyperparameter tuning using GridSearchCV

Cross-validation for time-series data

Deployment using Flask / FastAPI

Real-time stock API integration

LSTM comparison model

Streamlit dashboard for interactive predictions

Contributing

Contributions are welcome and appreciated.

To contribute:

Fork the project

Create your Feature Branch

git checkout -b feature/AmazingFeature

Commit your Changes

git commit -m "Add Amazing Feature"

Push to the Branch

git push origin feature/AmazingFeature

Open a Pull Request

License

Distributed under the MIT License.
See LICENSE for more information.

Authors

Anees Ahmad
GitHub: https://github.com/IaM-AnEeS

Project Link:
https://github.com/ayesha-aniqa/Stock_price_prediction

Acknowledgements

Scikit-learn Documentation

XGBoost Documentation

Kaggle (for financial data inspiration)

Open-source ML community

