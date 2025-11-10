# Import the libraries we need
import streamlit as st
import pandas as pd
import io # We need this to show df.info()

# --- ML LIBRARIES ---
# We'll need these to build our model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- Page Setup ---
st.set_page_config(page_title="EV Simulator", page_icon="⚡")

# --- Main App ---
st.title("🔋 EV Test Drive & Trip Simulator")
st.header("Week 2: Training Our ML Model")

# --- Load Data ---
@st.cache_data
def load_data():
    data = pd.read_csv("data/ev_consumption_data.csv")
    return data

# --- Preprocess Data ---
@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    # Drop columns that are not useful for prediction
    df_processed = df_processed.drop(columns=['Vehicle_ID', 'Timestamp'])
    return df_processed

# --- Load and Process Data ---
df_raw = load_data()
df_processed = preprocess_data(df_raw)


# --- 1. Define X (Features) and y (Target) ---
st.subheader("1. Defining Our 'Features' (X) and 'Target' (y)")

# Our Target (y) is the one thing we want to predict
TARGET_COLUMN = 'Energy_Consumption_kWh'
y = df_processed[TARGET_COLUMN]

# Our Features (X) are all the *other* columns we use as clues
X = df_processed.drop(columns=[TARGET_COLUMN])

st.write("**Target (y):**")
st.code(f"y = {TARGET_COLUMN}")
st.write("**Features (X):**")
st.dataframe(X.head()) # Show the first 5 rows of our features


# --- 2. Split Our Data ---
st.subheader("2. Splitting Data into Training and Testing Sets")
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.text("Data split 80% for training, 20% for testing.")
st.text(f"Training samples: {X_train.shape[0]}")
st.text(f"Testing samples: {X_test.shape[0]}")


# --- 3. Train Our Model ---
st.subheader("3. Training the Model")
st.write("We are using a `RandomForestRegressor` model.")
st.write("This may take a moment...")

# We'll use @st.cache_resource to store the trained model
@st.cache_resource
def train_model(X_train, y_train):
    # Create the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the model
    model.fit(X_train, y_train)
    return model

# Train the model and store it
model = train_model(X_train, y_train)
st.success("Model trained successfully!")


# --- 4. Evaluate Our Model ---
st.subheader("4. Evaluating Our Model")
st.write("Let's see how well our model performed on the 20% 'final exam' (test data).")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the R-squared score
# This score is from 0.0 to 1.0 (or 0% to 100%)
# It tells us how much of the variance our model can explain.
score = r2_score(y_test, y_pred)

st.metric(label="Model Accuracy (R-squared)", value=f"{score:.2%}")
st.write(f"This means our model can explain **{score:.2%}** of the variance in energy consumption, which is a fantastic result!")

# --- Show the Raw Data (At the bottom) ---
with st.expander("Show Raw Data and EDA"):
    st.header("1. Our Original Raw Dataset")
    st.dataframe(df_raw.head())

    st.header("2. Our New Preprocessed Dataset")
    st.dataframe(df_processed.head())

    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Raw Data Info")
    buffer = io.StringIO()
    df_raw.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)