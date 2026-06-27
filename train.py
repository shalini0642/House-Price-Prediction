from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics
joblib.dump(
    {
        "r2": r2,
        "mae": mae,
        "feature_names": list(X.columns)
    },
    "metrics.pkl"
)

print("Model and metrics saved!")
# Save predictions
joblib.dump(
    {
        "actual": y_test.values,
        "predicted": y_pred
    },
    "predictions.pkl"
)