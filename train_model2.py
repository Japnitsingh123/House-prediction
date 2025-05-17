import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("train.csv")

# Select features and target
features = ['YearBuilt', 'OverallQual', 'GarageCars', 'TotRmsAbvGrd',
            'Fireplaces', 'LotArea', 'HouseStyle', 'CentralAir']
target = 'SalePrice'

X = df[features].copy()
y = df[target]

# Encode categorical features
le_house_style = LabelEncoder()
X['HouseStyle'] = le_house_style.fit_transform(X['HouseStyle'])

le_central_air = LabelEncoder()
X['CentralAir'] = le_central_air.fit_transform(X['CentralAir'])

# Save encoders
joblib.dump(le_house_style, "HouseStyle_labelencoder.pkl")
joblib.dump(le_central_air, "CentralAir_labelencoder.pkl")

# Split into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and features
joblib.dump(scaler, "houseprice_scaler.pkl")
joblib.dump(features, "houseprice_features.pkl")

# Train model
model = xgb.XGBRegressor()
model.fit(X_train_scaled, y_train)

# Save the model
model.save_model("xgb_model.json")
