import os
import subprocess
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# Generate model files if they don't exist
# --------------------------------------------------
required_files = [
    "model.pkl",
    "metrics.pkl",
    "predictions.pkl"
]

if not all(os.path.exists(file) for file in required_files):
    with st.spinner("Training model for the first time..."):
        subprocess.run(["python", "train.py"], check=True)

# --------------------------------------------------
# Load model and data
# --------------------------------------------------
model = joblib.load("model.pkl")
metrics = joblib.load("metrics.pkl")
predictions = joblib.load("predictions.pkl")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🏠 House Price Prediction")

st.markdown("""
Predict California house prices using a **Machine Learning** model trained with
**Linear Regression**.
""")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("🏡 House Details")

MedInc = st.sidebar.slider(
    "Median Income", 0.0, 15.0, 3.0, 0.1,
    help="Median income in the block."
)

HouseAge = st.sidebar.slider(
    "House Age", 1.0, 60.0, 20.0, 1.0
)

AveRooms = st.sidebar.slider(
    "Average Rooms", 1.0, 15.0, 5.0, 0.1
)

AveBedrms = st.sidebar.slider(
    "Average Bedrooms", 0.5, 5.0, 1.0, 0.1
)

Population = st.sidebar.slider(
    "Population", 100, 10000, 1000, 100
)

AveOccup = st.sidebar.slider(
    "Average Occupancy", 1.0, 10.0, 3.0, 0.1
)

Latitude = st.sidebar.slider(
    "Latitude", 32.0, 42.0, 34.0, 0.1
)

Longitude = st.sidebar.slider(
    "Longitude", -125.0, -114.0, -118.0, 0.1
)

# --------------------------------------------------
# Model Metrics
# --------------------------------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric("R² Score", f"{metrics['r2']:.3f}")
col2.metric("Mean Absolute Error", f"{metrics['mae']:.3f}")

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.subheader("💰 Predict House Price")

if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude
        ]],
        columns=metrics["feature_names"]
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"🏡 Estimated House Price: **${prediction * 100000:,.2f}**"
    )

st.divider()

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("📈 Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    predictions["actual"],
    predictions["predicted"],
    alpha=0.5
)

min_val = min(predictions["actual"])
max_val = max(predictions["actual"])

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Ideal Prediction"
)

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted House Prices")
ax.legend()

st.pyplot(fig)

st.divider()

# --------------------------------------------------
# About
# --------------------------------------------------
st.subheader("ℹ️ About This Project")

st.markdown("""
### Technologies Used

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn
- 🌐 Streamlit
- 💾 Joblib

### Dataset

California Housing Dataset from **Scikit-learn**.

### Model

Linear Regression

This application predicts California house prices based on eight housing features.
""")