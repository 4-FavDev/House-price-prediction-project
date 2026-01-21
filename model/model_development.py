# ================================
# HOUSE PRICE PREDICTION MODEL
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import os

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("data/train.csv")


# -------------------------------
# Feature Selection (STRICTLY ALLOWED)
# -------------------------------
features = [
    'OverallQual',
    'GrLivArea',
    'TotalBsmtSF',
    'GarageCars',
    'BedroomAbvGr',
    'YearBuilt'
]

X = data[features]
y = data['SalePrice']

# -------------------------------
# Handle Missing Values
# -------------------------------
X = X.fillna(X.mean())

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training (Random Forest)
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MODEL EVALUATION RESULTS")
print("------------------------")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# -------------------------------
# Save Model
# -------------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_price_model.pkl")

print("\nModel saved successfully!")

# -------------------------------
# Reload Model (Verification)
# -------------------------------
loaded_model = joblib.load("model/house_price_model.pkl")

# Test reload
test_prediction = loaded_model.predict(X_test.iloc[:1])
print("Model reloaded and working correctly.")
