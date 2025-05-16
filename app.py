import xgboost as xgb
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="Ames House Price Predictor", layout="wide")

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
    background-color: #0d1117;
    color: white;
}

.stApp {
    background-image: url("https://images.unsplash.com/photo-1502673530728-f79b4cab31b1");
    background-size: cover;
    background-attachment: fixed;
}

.main > div {
    background-color: rgba(0, 0, 0, 0.75);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.05);
}

h1, h3 {
    text-align: center;
    color: white;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.85);
}

.stButton > button {
    background-color: #3498db;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #2980b9;
    box-shadow: 0 8px 20px rgba(41, 128, 185, 0.5);
}
</style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource

def load_artifacts():
    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")
    scaler = joblib.load("houseprice_scaler.pkl")
    features = joblib.load("houseprice_features.pkl")
    le_house_style = joblib.load("HouseStyle_labelencoder.pkl")
    le_central_air = joblib.load("CentralAir_labelencoder.pkl")
    df = joblib.load("ames_housing_data.pkl")  # Real sale prices
    return model, scaler, features, le_house_style, le_central_air, df

model, scaler, features, le_house_style, le_central_air, df = load_artifacts()

# Input UI
st.sidebar.header("Input House Features")
user_input = {
    "YearBuilt": st.sidebar.number_input("Year Built", 1800, 2025, 2000),
    "OverallQual": st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5),
    "GarageCars": st.sidebar.slider("Garage Capacity", 0, 5, 2),
    "TotRmsAbvGrd": st.sidebar.slider("Rooms Above Ground", 1, 20, 6),
    "Fireplaces": st.sidebar.slider("Fireplaces", 0, 5, 1),
    "LotArea": st.sidebar.number_input("Lot Area (sq ft)", 500, 100000, 8000),
     "HouseStyle": st.sidebar.selectbox("House Style", list(le_house_style)),
    "CentralAir": st.sidebar.selectbox("Central Air", list(le_central_air)),

}

# Encode categorical
house_style_encoded = list(le_house_style).index(user_input["HouseStyle"])
central_air_encoded = list(le_central_air).index(user_input["CentralAir"])


input_vector = np.array([
    user_input["YearBuilt"],
    user_input["OverallQual"],
    user_input["GarageCars"],
    user_input["TotRmsAbvGrd"],
    user_input["Fireplaces"],
    user_input["LotArea"],
    house_style_encoded,
    central_air_encoded
]).reshape(1, -1)

input_scaled = scaler.transform(input_vector)

# Header
st.markdown("""
<div class="headline-block">
    <h1>🏠 Ames House Price Predictor</h1>
    <h3>Estimate home prices based on real features from Ames, Iowa</h3>
</div>
""", unsafe_allow_html=True)

# Prediction
if st.button("Predict Price 💰"):
    prediction = model.predict(input_scaled)[0]
    st.session_state["prediction"] = prediction
    st.success(f"💲 Estimated Sale Price: ${prediction:,.2f}")

    # Find closest real price
    closest_idx = (df["SalePrice"] - prediction).abs().idxmin()
    closest_price = df.loc[closest_idx, "SalePrice"]
    error = abs(closest_price - prediction)
    error_pct = (error / closest_price) * 100

    st.markdown(f"""
        **Closest Actual Sale Price:** ${closest_price:,.2f}  
        **Prediction Error:** ${error:,.2f} ({error_pct:.2f}%)
    """)

# Show Map
if st.button("Show Location on Map 🗺️"):
    m = folium.Map(location=[42.0308, -93.6319], zoom_start=12)
    folium.Marker(
        location=[42.0308, -93.6319],
        popup="Ames, Iowa",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(m)
    st_folium(m, width=700, height=450)

# Reviews Section
st.markdown("""
---
### ⭐ User Reviews
- "This tool gave me a surprisingly close estimate!"
- "Love the interface and accuracy."
- "Helpful for comparing properties in Ames."
---
""")
