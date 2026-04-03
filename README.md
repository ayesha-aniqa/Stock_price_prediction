# 📈 NVIDIA Stock Price Prediction (XGBoost)

An end-to-end Machine Learning solution for predicting NVIDIA's (NVDA) stock price trends. This project features a consolidated production pipeline and a clean, pastel-themed Streamlit dashboard.

[![Fellowship](https://img.shields.io/badge/GDGOC%20Attock-AIML--Fellowship1-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](#)
[![Framework](https://img.shields.io/badge/Model-XGBoost-orange)](#)

---

## 📌 Table of Contents
* [Project Workflow](#-project-workflow)
* [Visuals & Flow Diagram](#-visuals--flow-diagram)
* [Repository Structure](#-repository-structure)
* [Installation & Usage](#-installation--usage)
* [Model Performance & Metrics](#-model-performance--metrics)
* [Model Limitations](#-model-limitations)
* [Contributions](#-contributions)

---

## ⚙️ Project Workflow
The project is designed with a **"Rule of One"** philosophy—consolidating complex logic into a single, high-performance engine.

1.  **Data Preprocessing**: Historical data is cleaned, sorted by date, and handled for null values.
2.  **Feature Engineering**: Technical indicators including Moving Averages ($MA_{10}, MA_{50}$), Lag features ($Lag_1, Lag_2, Lag_3$), and Volatility are calculated.
3.  **Data Scaling**: A `MinMaxScaler` is fitted strictly on training data to prevent data leakage.
4.  **Model Training**: An XGBoost Regressor is trained to predict the next day's closing price.
5.  **Deployment**: A Streamlit UI provides an interface to input market values and receive instant predictions.

---

## 📊 Visuals & Flow Diagram

### Project Flowchart
![Project Flowchart](reports/flowchart.png)

### Model Insights
The pipeline automatically generates and saves the following reports in the `reports/` directory:
* **Feature Importance**: Visualizes which technical indicators drive the prediction.
* **Actual vs Predicted**: A time-series comparison of model performance.
* **Residual Distribution**: Analysis of prediction errors.

---

## 📂 Repository Structure
```text
Stock_price_prediction/
├── app/
│   └── streamlit_app.py      # Pastel-themed UI
├── data/
│   ├── nvidia.csv            # Raw historical data
│   └── processed_nvidia.csv  # Engineered features
├── models/
│   ├── xgboost_model.pkl     # Trained engine
│   └── scaler.pkl            # Saved normalization parameters
├── notebooks/
│   └── stock_price_prediction.ipynb # Research & Experimentation
├── reports/
│   └── (Auto-generated charts and visuals)
├── src/
│   └── pipeline.py           # The "Master Script" for all logic
├── requirements.txt          # Environment dependencies
└── README.md
```
## Built With

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Pickle

## Getting Started

To run this project locally, follow the instructions below.



## Dependencies

Required libraries:

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

Install all dependencies using:
```bash
pip install -r requirements.txt
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Installation

Clone the repository:
```bash
git clone https://github.com/ayesha-aniqa/Stock_price_prediction
cd Stock_price_prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Navigate to the project folder:


## Usage

The project is organized into Jupyter notebooks representing each stage of the machine learning pipeline.

### 1. Dataset
kaggle Dataset link: https://www.kaggle.com/datasets/amirhoseinmousavian/nvidia-corporation-nvda-stock-price

## Description
This dataset is a time-series stock market dataset that contains daily trading information for a financial asset from 2020 to 2024. Each row represents a single trading day and includes key features such as the opening price, highest and lowest prices during the day, closing price, adjusted closing price (which accounts for dividends and stock splits), and trading volume. This type of dataset is commonly used for analyzing market trends, studying price movements, and building predictive models in finance.

Each row represents one trading day, with the following columns:

Date – the trading day

Open – the price at which the asset opened

High – the highest price reached during the day

Low – the lowest price during the day

Close – the price at market close

Adj Close – the adjusted closing price (accounts for dividends, splits, etc.)

Volume – the number of shares traded that day
Overall, the dataset is useful for financial analysis, such as tracking price trends, performing technical analysis, building predictive models, or studying market behavior over time.

### 2. Data Preprocessing

Run: 
```bash
notebooks/preprocessing.ipynb
```

This notebook:
- Cleans missing values
- Creates the target variable
- Splits dataset into training and testing sets (90% train, 10% test)
- Applies MinMax scaling

### 3. Model Training

Run: 
```bash
notebooks/model_training.ipynb
```

This notebook:
- Trains an XGBoost Regressor
- Uses hyperparameters:
  - n_estimators = 500
  - learning_rate = 0.03
  - max_depth = 6
  - random_state = 42
- Saves trained model as:
  - xGboost_model.pkl
  - scaler.pkl

### 4. Model Testing

Run: notebooks/model_testing.ipynb


This notebook:
- Loads saved model and scaler
- Predicts stock prices
- Evaluates performance using:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score

### 5. Model Visualization

Run: notebooks/Model_Visualization.ipynb


This notebook generates:

- Actual vs Predicted Stock Price Plot
- Feature Importance Graph
- Residual Error Distribution
- Scatter Plot (Predicted vs Actual)
- Model Performance Summary

### 6. Visuals

![Actual vs Predicted](assets/Actual_Predicted.png)
![Feature Importance](assets/Feature_Importance_XGBoost.png)
![Model Performance](assets/Model_Performance_Metrics.png)
![Residuals](assets/Residual_Error_Distribution.png)
![Scatter Plot](assets/Scatter_Plot.png)

## Project Workflow

![flowchart](https://github.com/user-attachments/assets/e6b40f5b-65eb-498c-bc34-7a63a5a277b0)

This structured workflow ensures reproducibility, clarity, and professional implementation standards.

---

## Results

The model performance is evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

The XGBoost model demonstrates strong predictive capability in modeling nonlinear patterns in stock price data.

Visual comparisons between actual and predicted values show close alignment, validating model effectiveness.

## Roadmap

Future improvements:

- Hyperparameter tuning using GridSearchCV
- Time-series cross-validation
- Real-time stock API integration
- Deployment using Flask or Streamlit
- Comparison with Deep Learning models (LSTM/GRU)

## Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a feature branch:
3. Commit changes:
4. Push to branch:
5. Open a Pull Request

---

## Team Contribuition

## 👥 Team
| Member | Role | Contribution |
|--------|------|-------------|
| [Ayesha Aniqa](https://github.com/ayesha-aniqa) | Team Lead | Team leadership, website frontend, model evaluation, performance analysis, collaboration & coordination |
| [Anees Ahmad](https://github.com/IaM-AnEeS) | ML Engineer | Dataset preprocessing, data cleaning, feature engineering, model training & development, README contribution |
| [Kashan Saqib](https://github.com/Kashhan) | QA Engineer | Model testing, error analysis, validation & performance testing |
| [Muhammad Mahaz Noor](https://github.com/mahaznoor) | Technical Writer | Documentation, README preparation, technical writing, content structuring |
| [Hizar Abdullah](https://github.com/khizerista) | Data Analyst | Model visualization, data visualization, result visualization, graphical representation |




## Acknowledgements

- Scikit-learn Documentation
- XGBoost Official Documentation
- Kaggle (Dataset Source)
- Open Source Community


If you found this project useful, consider giving it a star.
Thank You

---

## License

Distributed under the MIT License.  
See LICENSE file for more information.
