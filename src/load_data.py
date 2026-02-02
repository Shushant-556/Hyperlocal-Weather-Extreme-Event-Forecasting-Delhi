import pandas as pd

# Load dataset
file_path = "data/delhi_weather_pollution.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape (rows, columns):", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)
