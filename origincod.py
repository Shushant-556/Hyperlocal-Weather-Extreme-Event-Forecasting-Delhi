import streamlit as st
import numpy as np
import joblib
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Delhi Hyperlocal Weather Intelligence",
    page_icon="🌍",
    layout="wide"
)

# ==================================================
# DARK-ONLY PREMIUM CSS (NO WHITE ANYWHERE)
# ==================================================
st.markdown("""
<style>

/* -------- GLOBAL -------- */
html, body, [class*="css"] {
    background-color: #020617;
    color: #c7d2fe;
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* -------- HERO -------- */
.hero {
    padding: 70px 50px;
    background: linear-gradient(160deg, #020617, #020617, #0f172a);
    border-radius: 30px;
    margin-bottom: 50px;
    border: 1px solid #1e293b;
}

.hero h1 {
    font-size: 54px;
    font-weight: 800;
    background: linear-gradient(to right, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 18px;
    color: #94a3b8;
    max-width: 900px;
}

/* -------- GLASS CARD -------- */
.glass {
    background: linear-gradient(160deg, #020617, #0f172a);
    border-radius: 22px;
    padding: 30px;
    border: 1px solid #1e293b;
    box-shadow: 0 25px 50px rgba(0,0,0,0.8);
    margin-bottom: 35px;
}

/* -------- METRICS -------- */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #020617, #0f172a);
    border-radius: 16px;
    padding: 18px;
    border: 1px solid #1e293b;
    color: #e0e7ff;
}

/* -------- BUTTON -------- */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #22d3ee, #6366f1);
    color: #020617;
    border-radius: 18px;
    height: 60px;
    font-size: 18px;
    font-weight: 700;
    border: none;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* -------- BADGES -------- */
.badge-green {
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    padding: 14px 20px;
    border-radius: 16px;
    font-size: 18px;
    border: 1px solid #166534;
}

.badge-orange {
    background: rgba(249,115,22,0.15);
    color: #fdba74;
    padding: 14px 20px;
    border-radius: 16px;
    font-size: 18px;
    border: 1px solid #9a3412;
}

.badge-red {
    background: rgba(239,68,68,0.18);
    color: #fca5a5;
    padding: 14px 20px;
    border-radius: 16px;
    font-size: 18px;
    border: 1px solid #7f1d1d;
}

/* -------- CONTACT -------- */
.contact-card {
    background: linear-gradient(160deg, #020617, #020617, #0f172a);
    border-radius: 26px;
    padding: 40px;
    border: 1px solid #1e293b;
    box-shadow: 0 30px 70px rgba(0,0,0,0.9);
    margin-top: 60px;
}

.contact-title {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(to right, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 18px;
}

.contact-sub {
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 30px;
    max-width: 900px;
}

.contact-item {
    font-size: 18px;
    margin-bottom: 14px;
    color: #c7d2fe;
}

.contact-item span {
    color: #22d3ee;
    font-weight: 600;
}

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    color: #64748b;
    margin: 40px 0 10px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODELS
# ==================================================
pm25_model = joblib.load("models/pm25_model.pkl")
pm10_model = joblib.load("models/pm10_model.pkl")
extreme_model = joblib.load("models/extreme_pollution_classifier.pkl")
fog_model = joblib.load("models/fog_prediction_model.pkl")

# ==================================================
# API CONFIG (SECURE)
# ==================================================
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 28.6139, 77.2090

def fetch_real_time_data():
    if API_KEY is None:
        raise ValueError("API key not found")

    weather = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    ).json()

    air = requests.get(
        f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    ).json()

    return {
        "temperature": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
        "pm25": air["list"][0]["components"]["pm2_5"],
        "pm10": air["list"][0]["components"]["pm10"],
    }

# ==================================================
# HERO
# ==================================================
st.markdown("""
<div class="hero">
    <h1>Delhi Hyperlocal Weather Intelligence</h1>
    <p>
        A dark-mode, AI-powered environmental intelligence platform delivering
        real-time pollution forecasts, fog risk analysis, and extreme air-quality alerts
        for Delhi NCR.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# LIVE CONDITIONS
# ==================================================
try:
    data = fetch_real_time_data()
except Exception:
    st.warning("Live data unavailable. Using fallback values.")
    data = {
        "temperature": 20.0,
        "humidity": 60.0,
        "pressure": 1010.0,
        "wind_speed": 2.0,
        "pm25": 150.0,
        "pm10": 200.0,
    }

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🌍 Live Environmental Snapshot")

c1, c2, c3 = st.columns(3)
c1.metric("🌡️ Temperature (°C)", f"{data['temperature']:.2f}")
c2.metric("💧 Humidity (%)", data["humidity"])
c3.metric("🌀 Wind Speed (m/s)", data["wind_speed"])

c4, c5, c6 = st.columns(3)
c4.metric("📊 Pressure (hPa)", data["pressure"])
c5.metric("🫁 PM2.5 (µg/m³)", data["pm25"])
c6.metric("🫁 PM10 (µg/m³)", data["pm10"])

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# INPUT + FORECAST
# ==================================================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🧪 Scenario Simulation Panel")

colA, colB = st.columns(2)

with colA:
    temperature = st.number_input("🌡️ Temperature (°C)", value=float(data["temperature"]))
    humidity = st.number_input("💧 Humidity (%)", value=float(data["humidity"]))
    pressure = st.number_input("📊 Pressure (hPa)", value=float(data["pressure"]))

with colB:
    wind_speed = st.number_input("🌀 Wind Speed (m/s)", value=float(data["wind_speed"]))
    pm25 = st.number_input("🫁 PM2.5 (µg/m³)", value=float(data["pm25"]))
    pm10 = st.number_input("🫁 PM10 (µg/m³)", value=float(data["pm10"]))

if st.button("🚀 Generate Hyperlocal Forecast"):

    pm25_pred = pm25_model.predict([[temperature, humidity, pressure, wind_speed, pm25, pm25]])[0]
    pm10_pred = pm10_model.predict([[temperature, humidity, pressure, wind_speed, pm10, pm10]])[0]
    risk = extreme_model.predict([[temperature, humidity, pressure, wind_speed, pm10]])[0]
    fog = fog_model.predict([[temperature, humidity, pressure, wind_speed, pm25, pm10]])[0]

    st.subheader("📈 Forecast Intelligence")

    a, b = st.columns(2)
    a.metric("Predicted PM2.5 (µg/m³)", f"{pm25_pred:.2f}")
    b.metric("Predicted PM10 (µg/m³)", f"{pm10_pred:.2f}")

    if risk == 0:
        st.markdown('<div class="badge-green">🟢 Normal Pollution Levels</div>', unsafe_allow_html=True)
    elif risk == 1:
        st.markdown('<div class="badge-orange">🟠 High Pollution Alert</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-red">🔴 Extreme Pollution Emergency</div>', unsafe_allow_html=True)

    if fog == 1:
        st.markdown('<div class="badge-red">🌫️ Fog Likely – Visibility Risk</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-green">✅ No Fog Expected</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# CONTACT
# ==================================================
st.markdown("""
<div class="contact-card">
    <div class="contact-title">📬 Contact the Owner</div>
    <div class="contact-sub">
        Hyperlocal AI-based atmospheric intelligence system developed for
        urban environmental risk assessment and forecasting.
    </div>

    <div class="contact-item">👤 <span>Developer:</span> Shushant Tiwari</div>
    <div class="contact-item">🎓 <span>Domain:</span> AI, Data Science & Environmental Analytics</div>
    <div class="contact-item">📍 <span>Location:</span> Delhi NCR, India</div>
    <div class="contact-item">📧 <span>Email:</span> shushant.tiwari@example.com</div>
    <div class="contact-item">🔗 <span>LinkedIn:</span> linkedin.com/in/shushanttiwari</div>
    <div class="contact-item">💻 <span>GitHub:</span> github.com/shushanttiwari</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    © 2026 Hyperlocal Weather Intelligence Platform • Delhi NCR
</div>
""", unsafe_allow_html=True)