import pandas as pd

# Load dataset
df = pd.read_csv("data/delhi_weather_pollution.csv")

# Convert timestamp to datetime
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

# -------------------------
# FOG LOGIC
# -------------------------
# Fog conditions (Delhi specific, explainable in viva)
# High humidity + very low wind speed + high pollution
df["fog"] = (
    (df["humidity"] >= 85) &
    (df["wind_speed"] <= 2) &
    (df["pm25"] >= 100)
).astype(int)

# -------------------------
# EXTREME EVENT LOGIC
# -------------------------
def extreme_event(row):
    if row["pm25"] >= 300:
        return "Extreme Pollution"
    elif row["temperature"] >= 45:
        return "Heatwave"
    elif row["fog"] == 1 and row["humidity"] >= 90:
        return "Extreme Fog"
    else:
        return "Normal"

df["extreme_event"] = df.apply(extreme_event, axis=1)

# -------------------------
# Quick verification
# -------------------------
print("New columns added successfully!")
print(df[["temperature", "humidity", "pm25", "fog", "extreme_event"]].head())

print("\nFog value counts:")
print(df["fog"].value_counts())

print("\nExtreme event distribution:")
print(df["extreme_event"].value_counts())
