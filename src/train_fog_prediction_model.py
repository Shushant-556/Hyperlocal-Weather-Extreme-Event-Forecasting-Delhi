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
# 3. Create fog label using domain rules
# --------------------------------------------------
def is_fog(row):
    if (
        row["humidity"] >= 90 and
        row["wind_speed"] <= 2 and
        row["temperature"] <= 15 and
        row["pm25"] >= 100
    ):
        return 1  # Fog
    else:
        return 0  # No Fog

df["fog"] = df.apply(is_fog, axis=1)

# --------------------------------------------------
# 4. Feature selection
# --------------------------------------------------
features = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "pm25",
    "pm10"
]

X = df[features]
y = df["fog"]

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
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------
y_pred = model.predict(X_test)

print("Fog Prediction Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Fog", "Fog"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# --------------------------------------------------
# 8. Save model locally
# --------------------------------------------------
joblib.dump(model, "models/fog_prediction_model.pkl")
print("Fog prediction model saved locally.")
