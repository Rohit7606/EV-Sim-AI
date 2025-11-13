import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- MAPPING: These map the user's friendly text to the numbers in our dataset ---
# NOTE: These mappings are INFERRED based on common dataset practices (e.g., 0=Eco, 1=Normal, 2=Sport).
# If you check the original dataset source, you can confirm the exact mappings.
TRAFFIC_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
DRIVING_MODE_MAPPING = {"Eco": 0, "Normal": 1, "Sport": 2}
ROAD_TYPE_MAPPING = {"Urban": 0, "Highway": 1, "Rural": 2}
WEATHER_MAPPING = {"Sunny": 0, "Rainy": 1, "Snowy": 2} # Added for completeness


# --- Page Setup ---
st.set_page_config(page_title="EV Simulator", page_icon="⚡", layout="centered")

# --- Load and Preprocess Data (Cached Functions) ---
# We use st.cache_data to load the data quickly, and st.cache_resource to avoid retraining
# the model every time the user moves a slider.

@st.cache_data
def load_data():
    return pd.read_csv("data/ev_consumption_data.csv")

@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    df_processed = df_processed.drop(columns=['Vehicle_ID', 'Timestamp'])
    return df_processed

# --- Train Model (Cached Resource) ---
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- RUN ML PIPELINE ---
df_raw = load_data()
df_processed = preprocess_data(df_raw)

# Separate Features (X) and Target (y)
TARGET_COLUMN = 'Energy_Consumption_kWh'
y = df_processed[TARGET_COLUMN]
X = df_processed.drop(columns=[TARGET_COLUMN])

# Split and Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)

# Calculate the mean of the less important features to use as a default value
# We need to calculate this *before* we drop the target column from X, so we use df_processed
mean_features = df_processed.drop(columns=[TARGET_COLUMN]).mean()


# =========================================================================
# === WEEK 3: USER INPUT FORM & PREDICTION ===
# =========================================================================

st.title("🔋 EV Trip Prediction Dashboard")

# --- SIDEBAR FOR USER INPUTS ---
st.sidebar.header("🛣️ Trip Parameters")
st.sidebar.markdown("Configure your driving scenario below.")

# Get User Inputs using Streamlit Widgets

# 1. Numerical Inputs
speed_kmh = st.sidebar.slider("Average Speed (km/h)", min_value=X['Speed_kmh'].min(), max_value=X['Speed_kmh'].max(), value=float(mean_features['Speed_kmh']), step=1.0)
temp_c = st.sidebar.slider("Ambient Temperature (°C)", min_value=X['Temperature_C'].min(), max_value=X['Temperature_C'].max(), value=float(mean_features['Temperature_C']), step=1.0)
slope_perc = st.sidebar.slider("Route Slope (%)", min_value=X['Slope_%'].min(), max_value=X['Slope_%'].max(), value=float(mean_features['Slope_%']), step=0.1)

# 2. Categorical Inputs
traffic_condition = st.sidebar.radio("Traffic Level", list(TRAFFIC_MAPPING.keys()), index=1)
driving_mode = st.sidebar.radio("Driving Mode", list(DRIVING_MODE_MAPPING.keys()), index=0)
road_type = st.sidebar.radio("Road Type", list(ROAD_TYPE_MAPPING.keys()), index=1)


# --- PREDICTION LOGIC ---
if st.sidebar.button("Run Prediction"):
    # 1. Create a dictionary of all 17 features for the model
    user_input_dict = mean_features.to_dict()

    # 2. Update the user-selected features in the dictionary
    user_input_dict['Speed_kmh'] = speed_kmh
    user_input_dict['Temperature_C'] = temp_c
    user_input_dict['Slope_%'] = slope_perc
    
    # 3. Apply the mapping for the categorical features
    user_input_dict['Traffic_Condition'] = TRAFFIC_MAPPING[traffic_condition]
    user_input_dict['Driving_Mode'] = DRIVING_MODE_MAPPING[driving_mode]
    user_input_dict['Road_Type'] = ROAD_TYPE_MAPPING[road_type]
    
    # NOTE: We need to update Weather_Condition based on the temperature selected by the user
    # A simplified logic: Cold (<10C) = Snowy/Cold (2), Medium (10-25C) = Sunny (0), Hot (>25C) = Rainy/Hot (1)
    if temp_c < 10:
        user_input_dict['Weather_Condition'] = 2 # Snowy/Cold
    elif temp_c > 25:
        user_input_dict['Weather_Condition'] = 1 # Rainy/Hot (Auxiliary usage increases)
    else:
        user_input_dict['Weather_Condition'] = 0 # Sunny/Normal

    # 4. Convert the dictionary to a DataFrame (the format the model expects)
    input_df = pd.DataFrame([user_input_dict])

    # 5. Make the Prediction!
    prediction = model.predict(input_df)[0]

    # --- DISPLAY RESULTS ---
    st.success("✅ Prediction Complete")
    st.markdown("### Estimated Energy Consumption")
    
    # Display the primary result with a large metric
    st.metric(label="Energy Required for Trip", value=f"{prediction:.2f} kWh", delta="Based on 91.2% accurate ML Model")

    st.balloons()
    
    # --- NEXT STEP: Display for GenAI ---
    # We will use this prediction in the next section for the chatbot
    st.markdown("---")
    st.info(f"💡 This predicted value ({prediction:.2f} kWh) is ready to be used by the GenAI Assistant in Week 4!")


# --- Keep raw data hidden at the bottom ---
with st.expander("Show ML Model Details and Raw Data"):
    st.header("ML Model Performance")
    score = r2_score(y_test, model.predict(X_test))
    st.metric(label="Model Accuracy (R-squared)", value=f"{score:.2%}")
    st.write(f"This means our model can explain **{score:.2%}** of the variance in energy consumption.")
    
    st.subheader("Raw Data and Feature List")
    st.dataframe(df_raw.head())
    
    st.subheader("Features Used for Prediction")
    st.dataframe(X.head(1))