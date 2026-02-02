# PROJECT LOG – Hyperlocal Weather & Extreme Event Forecasting (Delhi)

This document records EVERYTHING completed so far so the project can be resumed anytime
without relying on chat history.

---

## PROJECT TITLE
Hyperlocal Weather & Extreme Event Forecasting – Delhi

---

## DATASET DETAILS

Dataset file name:
delhi_weather_pollution.csv

Dataset location (LOCAL ONLY):
Documents/Delhi_Hyperlocal_Weather_Project/data/delhi_weather_pollution.csv

Dataset type:
Combined historical weather + air pollution dataset for Delhi

Approximate size:
~422 MB

Reason dataset is NOT on GitHub:
- GitHub file size limit is 100 MB
- Best ML practice is to exclude datasets and include only code

Important columns used:
- event_timestamp
- temperature
- humidity
- pressure
- wind_speed
- pm25
- pm10

---

## PROJECT FOLDER STRUCTURE (CURRENT)

Delhi_Hyperlocal_Weather_Project/
│
├── src/
│   ├── load_data.py
│   ├── create_labels.py
│   ├── train_temperature_model.py
│   └── train_pm25_model.py
│
├── data/
│   └── delhi_weather_pollution.csv   (local only, ignored by Git)
│
├── models/
│   └── temperature_model.pkl         (local only, ignored by Git)
│
├── notebooks/
│
├── README.md
├── PROJECT_LOG.md
└── .gitignore

---

## FILES CREATED & THEIR PURPOSE

.gitignore
Purpose:
Prevents GitHub from tracking datasets, trained models, and temporary files.

Contents:
data/
data/*.csv
models/
models/*.pkl
__pycache__/
*.pyc

---

README.md
Purpose:
Explains project overview, features, tech stack, and structure for GitHub and evaluation.

---

load_data.py
Purpose:
Loads dataset and verifies data integrity and columns.

---

create_labels.py
Purpose:
Creates derived labels for:
- Fog events
- Extreme pollution events
- Heatwave conditions

Fog logic is based on:
- High humidity
- Low wind speed
- High PM2.5

---

train_temperature_model.py
Purpose:
Forecasts future temperature using lag-based prediction.

Important concept applied:
- Data leakage detected initially
- Fixed using time-based (t+1) prediction

Model used:
RandomForestRegressor

Typical performance:
MAE ≈ 2–3 °C
RMSE ≈ 3–4 °C

---

train_pm25_model.py
Purpose:
Forecasts PM2.5 concentration using:
- Weather variables
- Lagged PM2.5 values (pm25_lag_1, pm25_lag_2)

Model used:
RandomForestRegressor

Status:
CREATED – READY TO RUN (STEP 9)

---

## MODELS (LOCAL ONLY)

Stored inside:
models/

Current models:
- temperature_model.pkl
- pm25_model.pkl (will be created)

These are NOT pushed to GitHub (correct practice).

---

## GITHUB DETAILS

Repository name:
Hyperlocal-Weather-Extreme-Event-Forecasting-Delhi

Repository status:
- Clean commit history
- Dataset excluded
- Model files excluded
- Contains src/, README.md, .gitignore

---

## COMPLETED STEPS

- Python environment setup
- Project structure created
- Dataset downloaded and stored locally
- Git installed and configured
- GitHub repository created and connected
- .gitignore configured correctly
- README.md added and pushed
- Dataset and models excluded from GitHub
- Temperature forecasting model completed

---

## CURRENT STEP

STEP 9 – PM2.5 Forecasting Model

File:
src/train_pm25_model.py

Command to run:
python src/train_pm25_model.py

---

## UPCOMING STEPS (ROADMAP)

STEP 10 – PM10 forecasting model  
STEP 11 – Extreme pollution classification  
STEP 12 – Fog classification model  
STEP 13 – Streamlit dashboard  
STEP 14 – Final PPT and submission preparation

---

## IMPORTANT NOTES

- GitHub contains ONLY code and documentation
- Dataset and trained models remain local
- This file allows project continuation even if chat history is lost

END OF LOG

STEP 9 COMPLETED:
PM2.5 forecasting model trained using time-based split.
Final performance:
MAE = 36.98
RMSE = 63.13
Results are realistic due to high volatility of Delhi PM2.5 levels.

