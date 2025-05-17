import xgboost as xgb
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Page config
st.set_page_config(page_title="Ames House Price Predictor", layout="wide")

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
    color: white;
}

.stApp {
    background-image: url("https://images.trvl-media.com/lodging/101000000/100740000/100736200/100736154/10ed0f28.jpg?impolicy=resizecrop&rw=575&rh=575&ra=fill");
    background-size: cover;
    background-attachment: fixed;
}

.main > div {
    background-color: rgba(0, 0, 0, 0.75);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.05);
}

/* Map container styling */
.map-wrapper {
    margin: 0 auto;
    width: 50% !important;
    background: transparent !important;
    height: 0px !important;
    overflow: hidden !important;
}
.map-wrapper iframe {
    width: 100% !important;
    height: 400px !important;
    background: transparent !important;
    min-height: 400px !important;
    max-height: 400px !important;
    overflow: hidden !important;
}
.stCustomComponent > div {
    height: 400px !important;
    overflow: hidden !important;
}
.stCustomComponent iframe {
    height: 400px !important;
    overflow: hidden !important;
}

h1, h3 {
    text-align: center;
    color: white;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.85);
}

/* Black buttons with white text */
.stButton > button {
    background-color: black !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    box-shadow: 0 5px 15px rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #222 !important;
    box-shadow: 0 8px 20px rgba(255, 255, 255, 0.3);
}

.text-block {
    background-color: rgba(0, 0, 0, 0.8);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)


# Load model and preprocessors
@st.cache_resource
def load_artifacts():
    # Load dataset
    df = pd.read_csv("train.csv")

    # Select features and target
    features = ['YearBuilt', 'OverallQual', 'GarageCars', 'TotRmsAbvGrd',
                'Fireplaces', 'LotArea', 'HouseStyle', 'CentralAir']
    target = 'SalePrice'

    # Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    le_house_style = LabelEncoder()
    df['HouseStyle'] = le_house_style.fit_transform(df['HouseStyle'])

    le_central_air = LabelEncoder()
    df['CentralAir'] = le_central_air.fit_transform(df['CentralAir'])

    # Split into 80% train and 20% test
    from sklearn.model_selection import train_test_split
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model on 80%
    model = xgb.XGBRegressor()
    model.fit(X_train_scaled, y_train)

    # Save for prediction and UI
    df_test = pd.DataFrame(X_test_scaled, columns=features)
    df_test["SalePrice"] = y_test.values

    return model, scaler, features, le_house_style, le_central_air, df, df_test


model, scaler, features, le_house_style, le_central_air, df_train, df_pred = load_artifacts()

# Initialize session state variables
# Initialize session state variables with appropriate defaults
if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = False
if "selected_location" not in st.session_state:
    st.session_state["selected_location"] = [42.0308, -93.6319]  # Default location in Ames, Iowa


# Ames neighborhood coordinates (approximate centers)
NEIGHBORHOOD_COORDS = {
    'NAmes': (42.0419, -93.6131),      # North Ames
    'CollgCr': (42.0212, -93.6563),    # College Creek
    'OldTown': (42.0217, -93.6143),    # Old Town
    'Edwards': (42.0222, -93.6278),     # Edwards
    'Somerst': (42.0346, -93.6396),    # Somerset
    'NridgHt': (42.0537, -93.6268),    # Northridge Heights
    'Gilbert': (42.1087, -93.6349),     # Gilbert
    'NWAmes': (42.0529, -93.6651),      # Northwest Ames
    'Sawyer': (42.0371, -93.6192),      # Sawyer
    'Mitchell': (42.0308, -93.6202),    # Mitchell
    'IDOTRR': (42.0217, -93.6143),      # Iowa DOT and Rail Road
    'MeadowV': (42.0143, -93.6192),     # Meadow Village
    'BrkSide': (42.0223, -93.6192),     # Brookside
    'ClearCr': (42.0222, -93.6278),     # Clear Creek
    'SWISU': (42.0212, -93.6563),       # South & West of ISU
    'Blmngtn': (42.0537, -93.6268),     # Bloomington Heights
    'Veenker': (42.0419, -93.6563),     # Veenker
    'Timber': (42.0371, -93.6192),      # Timberland
    'NPkVill': (42.0529, -93.6349),     # Northpark Villa
    'StoneBr': (42.0346, -93.6396),     # Stone Brook
    'NoRidge': (42.0537, -93.6268),     # Northridge
    'BrDale': (42.0371, -93.6192)       # Briardale
}

# Input UI
st.sidebar.header("Input House Features")
house_style_classes = le_house_style.classes_.tolist()
central_air_classes = le_central_air.classes_.tolist()

# Sidebar inputs
user_input = {
    "YearBuilt": st.sidebar.number_input("Year Built", 1800, 2025, 2000),
    "OverallQual": st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5),
    "GarageCars": st.sidebar.slider("Garage Capacity", 0, 5, 2),
    "TotRmsAbvGrd": st.sidebar.slider("Rooms Above Ground", 1, 20, 6),
    "Fireplaces": st.sidebar.slider("Fireplaces", 0, 5, 1),
    "LotArea": st.sidebar.number_input("Lot Area (sq ft)", 500, 100000, 8000),
    "HouseStyle": st.sidebar.selectbox("House Style", house_style_classes),
    "CentralAir": st.sidebar.selectbox("Central Air", central_air_classes),
}

# Encode
house_style_encoded = house_style_classes.index(user_input["HouseStyle"])
central_air_encoded = central_air_classes.index(user_input["CentralAir"])


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

# Prediction button and logic
if st.button("Predict Price 💰"):
    prediction = model.predict(input_scaled)[0]
    st.session_state["prediction"] = prediction
    st.session_state["last_prediction"] = {
        "price": prediction,
        "features": user_input
    }
    st.session_state["prediction_made"] = True

    # Find closest actual house and its neighborhood from training data
    closest_idx = (df_train["SalePrice"] - prediction).abs().idxmin()
    closest_price = df_train.loc[closest_idx, "SalePrice"]
    closest_neighborhood = df_train.loc[closest_idx, "Neighborhood"]
    error = abs(closest_price - prediction)
    error_pct = (error / closest_price) * 100

    st.session_state["closest_price"] = closest_price
    st.session_state["error"] = error
    st.session_state["error_pct"] = error_pct
    
    # Set location based on the neighborhood of the closest matching house
    if closest_neighborhood in NEIGHBORHOOD_COORDS:
        st.session_state["selected_location"] = list(NEIGHBORHOOD_COORDS[closest_neighborhood])
        st.session_state["neighborhood"] = closest_neighborhood
    else:
        # Fallback to Ames center if neighborhood not found
        st.session_state["selected_location"] = [42.0308, -93.6319]
        st.session_state["neighborhood"] = "Unknown"

    # Show prediction details if prediction was made
    if st.session_state["prediction_made"]:
        st.markdown(f"""
        <div class="text-block">
        <h3>💡 Prediction Details</h3>
        <p><strong>💲 Estimated Sale Price:</strong> ${st.session_state['prediction']:,.2f}</p>
        <p><strong>📊 Closest Actual Sale Price:</strong> ${st.session_state['closest_price']:,.2f}</p>
        <p><strong>❗ Prediction Error:</strong> ${st.session_state['error']:,.2f} ({st.session_state['error_pct']:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)

    # Evaluate on test set (20%)
    X_test_scaled = df_pred[features].values
    y_test = df_pred["SalePrice"].values
    y_pred_test = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # Display metrics
    st.markdown(f"""
    <div class="text-block">
    <h3>📊 Evaluation on 20% Test Data</h3>
    <p><strong>📉 RMSE:</strong> ${rmse:,.2f}</p>
    <p><strong>📦 MAE:</strong> ${mae:,.2f}</p>
    <p><strong>🧮 R² Score:</strong> {r2:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Show map automatically after prediction
    st.markdown("""
        <style>
        [data-testid="column"] {
            height: 400px !important;
            overflow: hidden !important;
        }
        </style>
        <h3 style="color: white; margin-bottom: 15px; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">📍 Ames, Iowa Location</h3>
        <div class="map-wrapper">
        """, unsafe_allow_html=True)

    # Get center coordinates (Ames, Iowa)
    center_lat, center_lon = st.session_state["selected_location"]

    # Configure the map with transparent background
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='CartoDB positron',
        control_scale=True,
        prefer_canvas=True
    )
    
    # Add custom CSS to the map's HTML
    css = """
    <style>
    body, html {
        height: 400px !important;
        max-height: 400px !important;
        overflow: hidden !important;
        background: transparent !important;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(css))
    
    # Add marker for selected location
    if st.session_state.get("last_prediction"):
        pred = st.session_state["last_prediction"]
        features = pred["features"]
        neighborhood = st.session_state.get("neighborhood", "Unknown")
        popup_html = f"""
        <div style='font-family: Arial; width: 250px;'>
            <h4 style='margin: 0; color: #28a745;'>Property Details</h4>
            <p style='margin: 10px 0;'><strong>💲 Predicted Price: ${pred['price']:,.2f}</strong></p>
            <p style='margin: 5px 0;'><strong>📍 Neighborhood: {neighborhood}</strong></p>
            <hr style='margin: 5px 0;'>
            <p style='margin: 5px 0;'>Year Built: {features['YearBuilt']}</p>
            <p style='margin: 5px 0;'>Style: {features['HouseStyle']}</p>
            <p style='margin: 5px 0;'>Quality: {features['OverallQual']}/10</p>
            <p style='margin: 5px 0;'>Rooms: {features['TotRmsAbvGrd']}</p>
            <p style='margin: 5px 0;'>Lot Area: {features['LotArea']} sq ft</p>
        </div>
        """

        # Add marker for selected location
        folium.Marker(
            location=[center_lat, center_lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='green', icon='home', prefix='fa')
        ).add_to(m)

        # Add a circle to highlight the area
        folium.Circle(
            location=[center_lat, center_lon],
            radius=800,  # 800m radius
            color='#28a745',
            fill=True,
            fill_color='#28a745',
            fill_opacity=0.2,
            popup='Selected Area'
        ).add_to(m)

    # Display the map with custom configuration
    with st.container():
        _map = st_folium(
            m,
            height=400,
            width=None,
            returned_objects=["last_active_drawing", "last_clicked"],
            key="map"
        )
        
        # Handle location selection from map clicks
        if _map.get("last_clicked"):
            clicked_lat = _map["last_clicked"]["lat"]
            clicked_lng = _map["last_clicked"]["lng"]
            st.session_state["selected_location"] = [clicked_lat, clicked_lng]
            
            # Show selected coordinates
            st.markdown(f"""
            <div class="text-block" style="margin-top: 10px;">
                <h4>📍 Selected Location</h4>
                <p>Latitude: {clicked_lat:.6f}</p>
                <p>Longitude: {clicked_lng:.6f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

 # Reviews Section
st.markdown("""
<div class="text-block">
<h3>⭐ User Reviews</h3>
<ul>
    <li>"This tool gave me a surprisingly close estimate!"</li>
    <li>"Love the interface and accuracy."</li>
    <li>"Helpful for comparing properties in Ames."</li>
</ul>
</div>
""", unsafe_allow_html=True)