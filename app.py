import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------
# Load Files
# ----------------------------
model = joblib.load("model.pkl")
metrics = joblib.load("metrics.pkl")
predictions = joblib.load("predictions.pkl")

# ----------------------------
# Title
# ----------------------------
st.title("🏠 House Price Prediction")

st.write(
    """
Predict California house prices using a Machine Learning model trained with
Linear Regression.
"""
)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🏠 Enter House Details")

MedInc = st.sidebar.number_input("Median Income", value=3.0)
HouseAge = st.sidebar.number_input("House Age", value=20.0)
AveRooms = st.sidebar.number_input("Average Rooms", value=5.0)
AveBedrms = st.sidebar.number_input("Average Bedrooms", value=1.0)
Population = st.sidebar.number_input("Population", value=1000.0)
AveOccup = st.sidebar.number_input("Average Occupancy", value=3.0)
Latitude = st.sidebar.number_input("Latitude", value=34.0)
Longitude = st.sidebar.number_input("Longitude", value=-118.0)

# ----------------------------
# Metrics
# ----------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric("R² Score", f"{metrics['r2']:.3f}")
col2.metric("Mean Absolute Error", f"{metrics['mae']:.3f}")

# ----------------------------
# Prediction
# ----------------------------
st.subheader("💰 Predict House Price")

if st.button("Predict Price"):

    data = pd.DataFrame(
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

    prediction = model.predict(data)[0]

    st.success(
        f"🏡 Estimated House Price: **${prediction * 100000:,.2f}**"
    )

# ----------------------------
# Graph
# ----------------------------
st.divider()

st.subheader("📈 Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    predictions["actual"],
    predictions["predicted"],
    alpha=0.5
)

# Ideal prediction line
min_val = min(predictions["actual"])
max_val = max(predictions["actual"])

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2
)

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Model Predictions")

st.pyplot(fig)

# ----------------------------
# About
# ----------------------------
st.divider()

st.subheader("ℹ️ About")

st.markdown("""
- **Dataset:** California Housing Dataset
- **Model:** Linear Regression
- **Library:** Scikit-learn
- **Frontend:** Streamlit
""")