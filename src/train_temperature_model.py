import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/delhi_weather_pollution.csv")

# Convert timestamp and sort
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df = df.sort_values("event_timestamp")

# -------------------------
# Create LAG feature (real forecasting)
# -------------------------
df["temperature_next"] = df["temperature"].shift(-1)

# Drop last row (no future value)
df = df.dropna()

# -------------------------
# Select features and target
# -------------------------
features = [
    "humidity",
    "pressure",
    "wind_speed",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co"
]

X = df[features]
y = df["temperature_next"]

# -------------------------
# Train-test split (time-based)
# -------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# -------------------------
# Train model
# -------------------------
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Temperature Forecasting Model Trained!")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# -------------------------
# Save model
# -------------------------
joblib.dump(model, "models/temperature_model.pkl")
print("Model saved successfully.")
