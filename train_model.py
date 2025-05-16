import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("train.csv")

# Select features and target
features = [
    "YearBuilt", "OverallQual", "GarageCars", "TotRmsAbvGrd", 
    "Fireplaces", "LotArea", "HouseStyle", "CentralAir"
]
X = df[features]
y = df["SalePrice"]

# Label encode categories
X["HouseStyle"] = X["HouseStyle"].astype("category")
X["CentralAir"] = X["CentralAir"].astype("category")
house_style_classes = X["HouseStyle"].cat.categories
central_air_classes = X["CentralAir"].cat.categories
X["HouseStyle"] = X["HouseStyle"].cat.codes
X["CentralAir"] = X["CentralAir"].cat.codes

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = xgb.XGBRegressor()
model.fit(X_scaled, y)

# Save
joblib.dump(model, "xgb_houseprice_model.pkl")
joblib.dump(scaler, "houseprice_scaler.pkl")
joblib.dump(features, "houseprice_features.pkl")
joblib.dump(house_style_classes, "HouseStyle_labelencoder.pkl")
joblib.dump(central_air_classes, "CentralAir_labelencoder.pkl")
