import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("train.csv")

# Select features and target
features = ['YearBuilt', 'OverallQual', 'GarageCars', 'TotRmsAbvGrd', 'Fireplaces', 'LotArea', 'HouseStyle', 'CentralAir']
target = 'SalePrice'

X = df[features]
y = df[target]

# Label encode categorical columns
le_house_style = LabelEncoder()
X['HouseStyle'] = le_house_style.fit_transform(X['HouseStyle'])
joblib.dump(le_house_style, "HouseStyle_labelencoder.pkl")

le_central_air = LabelEncoder()
X['CentralAir'] = le_central_air.fit_transform(X['CentralAir'])
joblib.dump(le_central_air, "CentralAir_labelencoder.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and feature list
joblib.dump(scaler, "houseprice_scaler.pkl")
joblib.dump(features, "houseprice_features.pkl")

# Split dataset for training/testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost regression model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "xgb_houseprice_model.pkl")

print("Training complete. Model, scaler, and label encoders saved!")
