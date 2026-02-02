import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("data/delhi_weather_pollution.csv")

# --------------------------------------------------
# 2. Convert timestamp and sort (CRITICAL)
# --------------------------------------------------
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df = df.sort_values("event_timestamp").reset_index(drop=True)

# --------------------------------------------------
# 3. Create lag features for PM10
# --------------------------------------------------
df["pm10_lag_1"] = df["pm10"].shift(1)
df["pm10_lag_2"] = df["pm10"].shift(2)

# --------------------------------------------------
# 4. Drop missing rows
# --------------------------------------------------
df = df.dropna().reset_index(drop=True)

# --------------------------------------------------
# 5. Feature selection
# --------------------------------------------------
features = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "pm10_lag_1",
    "pm10_lag_2"
]

X = df[features]
y = df["pm10"]

# --------------------------------------------------
# 6. Time-based train/test split (NO leakage)
# --------------------------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# --------------------------------------------------
# 7. Train model
# --------------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("PM10 Forecasting Model Trained Successfully!")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# --------------------------------------------------
# 9. Save model locally
# --------------------------------------------------
joblib.dump(model, "models/pm10_model.pkl")
print("PM10 model saved locally.")
