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
# 3. Create lag features for PM2.5 (real forecasting)
# --------------------------------------------------
df["pm25_lag_1"] = df["pm25"].shift(1)
df["pm25_lag_2"] = df["pm25"].shift(2)

# --------------------------------------------------
# 4. Drop rows with missing values
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
    "pm25_lag_1",
    "pm25_lag_2"
]

X = df[features]
y = df["pm25"]

# --------------------------------------------------
# 6. TIME-BASED TRAIN / TEST SPLIT (NO LEAKAGE)
# --------------------------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# --------------------------------------------------
# 7. Train Random Forest model
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

print("PM2.5 Forecasting Model Trained Successfully!")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# --------------------------------------------------
# 9. Save model locally (NOT for GitHub)
# --------------------------------------------------
joblib.dump(model, "models/pm25_model.pkl")
print("PM2.5 model saved locally.")
