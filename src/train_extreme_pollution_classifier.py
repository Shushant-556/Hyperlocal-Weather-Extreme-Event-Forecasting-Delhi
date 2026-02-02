import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("data/delhi_weather_pollution.csv")

# --------------------------------------------------
# 2. Convert timestamp and sort
# --------------------------------------------------
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df = df.sort_values("event_timestamp").reset_index(drop=True)

# --------------------------------------------------
# 3. Create pollution risk labels
# --------------------------------------------------
def pollution_risk(pm25):
    if pm25 <= 60:
        return 0  # Normal
    elif pm25 <= 250:
        return 1  # High Pollution
    else:
        return 2  # Extreme Pollution

df["pollution_risk"] = df["pm25"].apply(pollution_risk)

# --------------------------------------------------
# 4. Feature selection
# --------------------------------------------------
features = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "pm10"
]

X = df[features]
y = df["pollution_risk"]

# --------------------------------------------------
# 5. Time-based train/test split
# --------------------------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# --------------------------------------------------
# 6. Train classifier
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Evaluate model (FIXED)
# --------------------------------------------------
y_pred = model.predict(X_test)

print("Extreme Pollution Classification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Normal", "High Pollution", "Extreme Pollution"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

# --------------------------------------------------
# 8. Save model locally
# --------------------------------------------------
joblib.dump(model, "models/extreme_pollution_classifier.pkl")
print("Extreme pollution classifier saved locally.")
