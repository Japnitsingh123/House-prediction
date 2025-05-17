import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

# Load data (replace with your actual test data file path)
df_test = pd.read_csv("train.csv")  # <-- Update if needed

# Load preprocessor and features
scaler = joblib.load("houseprice_scaler.pkl")
features = joblib.load("houseprice_features.pkl")
le_house_style = joblib.load("HouseStyle_labelencoder.pkl")
le_central_air = joblib.load("CentralAir_labelencoder.pkl")

# Prepare test data (make sure 'features' matches what the model was trained on)
df_encoded = df_test.copy()
df_encoded["HouseStyle"] = df_encoded["HouseStyle"].apply(lambda x: list(le_house_style).index(x))
df_encoded["CentralAir"] = df_encoded["CentralAir"].apply(lambda x: list(le_central_air).index(x))
X_test = scaler.transform(df_encoded[features])
y_test = df_encoded["SalePrice"]

# Load model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

# Predict
y_pred = model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test, y=y_test, color='red', label='Perfect Prediction')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
