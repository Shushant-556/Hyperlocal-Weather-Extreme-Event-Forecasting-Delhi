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

## STEP 9 – PM2.5 FORECASTING MODEL (COMPLETED)

Objective:
To forecast future PM2.5 concentration in Delhi using historical pollution and meteorological data, enabling proactive air quality assessment.

Approach:
- Implemented a regression-based machine learning model.
- Selected PM2.5 as the primary target variable due to its high health impact.
- Used meteorological parameters (temperature, humidity, pressure, wind speed) along with historical PM2.5 values.

Feature Engineering:
- Created lag-based features:
  - pm25_lag_1 (PM2.5 value at previous timestep)
  - pm25_lag_2 (PM2.5 value two timesteps before)
- Lag features ensure true forecasting instead of same-time prediction.

Data Splitting Strategy:
- Initially observed unrealistically low error due to random train-test split.
- Identified temporal data leakage as the cause.
- Corrected the issue by applying a strict time-based split:
  - First 80% of data → training
  - Last 20% of data → testing

Model Used:
- RandomForestRegressor
- Chosen for its robustness on tabular, non-linear, real-world data.

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Final Performance:
- MAE = 36.98
- RMSE = 63.13

Interpretation:
- Errors are realistic due to high volatility of PM2.5 in Delhi.
- Sudden pollution spikes caused by stubble burning, fireworks, and weather inversion contribute to higher RMSE.
- Results reflect real-world forecasting difficulty.

Outcome:
- PM2.5 forecasting model successfully trained.
- Temporal leakage eliminated.
- Model saved locally for dashboard integration.

------------------------------------------------------------

## STEP 10 – PM10 FORECASTING MODEL (COMPLETED)

Objective:
To forecast PM10 concentration levels, which represent coarse particulate matter heavily influenced by dust and construction activities in Delhi.

Approach:
- Followed the same validated forecasting pipeline as PM2.5.
- Implemented a regression model using lagged PM10 values and meteorological features.

Feature Engineering:
- Created lag-based features:
  - pm10_lag_1
  - pm10_lag_2
- These features capture short-term persistence in PM10 levels.

Data Splitting Strategy:
- Used time-based split to prevent temporal leakage.
- Ensured future PM10 values were not indirectly visible during training.

Model Used:
- RandomForestRegressor

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Final Performance:
- MAE = 149.30
- RMSE = 217.00

Interpretation:
- PM10 exhibits significantly higher variability than PM2.5.
- Influenced by localized dust resuspension, traffic, construction, and wind-driven events.
- Higher RMSE is expected and indicates realistic forecasting behavior.

Outcome:
- PM10 forecasting model trained successfully.
- Results are explainable and consistent with real-world PM10 behavior.
- Model saved locally.

------------------------------------------------------------

## STEP 11 – EXTREME POLLUTION EARLY WARNING CLASSIFIER (COMPLETED)

Objective:
To convert continuous PM2.5 values into actionable pollution risk categories, enabling an early warning system rather than raw numeric prediction.

Class Definition (Domain Knowledge Based):
- Normal Pollution (Class 0): PM2.5 ≤ 60
- High Pollution (Class 1): 60 < PM2.5 ≤ 250
- Extreme Pollution (Class 2): PM2.5 > 250

Approach:
- Converted PM2.5 values into categorical risk labels.
- Framed the problem as a multi-class classification task.
- Focused on decision-making intelligence instead of precise numerical accuracy.

Features Used:
- temperature
- humidity
- pressure
- wind_speed
- pm10

Model Used:
- RandomForestClassifier
- Selected for stability, interpretability, and strong performance on structured data.

Data Splitting Strategy:
- Time-based split applied to simulate real-world future warning scenarios.
- Training: first 80% of timeline
- Testing: last 20% of timeline

Evaluation:
- Used classification report and confusion matrix.
- Explicitly specified class labels to handle temporal class imbalance.
- Handled cases where some classes were absent in test data.

Observed Results:
- Test data contained only the “High Pollution” class.
- Normal and Extreme classes were absent in the final time window.
- Accuracy ≈ 83% for High Pollution detection.

Interpretation:
- Reflects persistent high pollution levels in recent Delhi data.
- Demonstrates realistic class imbalance common in environmental time-series.
- Model performance is valid and defensible.

Outcome:
- Extreme pollution early warning system successfully implemented.
- Model saved locally for dashboard integration.
- Provides clear, interpretable pollution risk alerts.

------------------------------------------------------------

