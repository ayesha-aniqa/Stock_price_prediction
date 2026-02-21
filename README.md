This project predicts NVIDIA (NVDA) stock prices using Machine Learning techniques. The model is built using historical stock market data and trained with the XGBoost algorithm. The project includes data preprocessing, model training, evaluation, model saving, and visualization of results.

The main goal is to analyze past stock price trends and build a predictive model that can estimate future stock prices with good accuracy.

🏢 Company Information
4

Company Name: NVIDIA
Stock Symbol: NVDA
Industry: Semiconductor & Artificial Intelligence
Stock Exchange: NASDAQ

🛠️ Technologies Used

Python

NumPy

Pandas

XGBoost

Matplotlib

Seaborn

Pickle

📊 Project Workflow
1️⃣ Data Preprocessing

Loaded historical NVIDIA stock dataset

Handled missing values

Selected relevant features

Split data into training and testing sets

2️⃣ Model Training

Trained the model using XGBoost Regressor

Optimized model performance

Reduced overfitting using regularization

3️⃣ Model Testing & Evaluation

Tested model on unseen data

Evaluated using MAE, MSE, and R² score

4️⃣ Model Saving

Saved trained model using Pickle for future use

5️⃣ Data Visualization

Visualized Actual vs Predicted prices

Analyzed stock trends

Plotted error distribution using Matplotlib and Seaborn