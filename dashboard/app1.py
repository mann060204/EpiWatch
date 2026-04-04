import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="EpiWatch", layout="wide")

# ==============================
# Load Data
# ==============================
data = pd.read_csv("data/owid-covid-data.csv")
data["date"] = pd.to_datetime(data["date"])

# ==============================
# 🔥 Train ML Model (FINAL SAFE)
# ==============================
@st.cache_resource
def train_model(data):

    df = data[[
        "total_cases",
        "total_deaths",
        "population",
        "new_cases"
    ]].copy()

    # Remove invalid rows
    df = df[
        (df["population"] > 0) &
        (df["total_cases"] > 0) &
        (df["new_cases"] >= 0)
    ]

    # Feature Engineering
    df["cases_per_million"] = (df["new_cases"] / df["population"]) * 1_000_000
    df["death_rate"] = df["total_deaths"] / df["total_cases"]

    # Clean data
    df.replace([float("inf"), -float("inf")], 0, inplace=True)
    df.fillna(0, inplace=True)

    # Clip extreme values
    df["cases_per_million"] = df["cases_per_million"].clip(0, 10000)
    df["death_rate"] = df["death_rate"].clip(0, 1)

    # Labels
    df["severity"] = 0
    df.loc[df["cases_per_million"] > 50, "severity"] = 1
    df.loc[df["cases_per_million"] > 150, "severity"] = 2

    # Features
    X = df[["cases_per_million", "death_rate", "new_cases"]]
    y = df["severity"]

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X, y)
    return model


model = train_model(data)

# ==============================
# Country Selector
# ==============================
country = st.selectbox(
    "Select Country",
    sorted(data["location"].dropna().unique())
)

df = data[data["location"] == country]

# ==============================
# Metrics
# ==============================
latest_cases = df["total_cases"].dropna().iloc[-1] if not df["total_cases"].dropna().empty else 0
latest_deaths = df["total_deaths"].dropna().iloc[-1] if not df["total_deaths"].dropna().empty else 0
latest_new_cases = df["new_cases"].dropna().iloc[-1] if not df["new_cases"].dropna().empty else 0
population = df["population"].dropna().iloc[-1] if not df["population"].dropna().empty else 0

# ==============================
# 🔥 ML Prediction (SAFE)
# ==============================
cases_per_million = (latest_new_cases / population) * 1_000_000 if population > 0 else 0
death_rate = latest_deaths / latest_cases if latest_cases > 0 else 0

# Safety clipping
cases_per_million = min(cases_per_million, 10000)
death_rate = min(death_rate, 1)

prediction = model.predict([
    [cases_per_million, death_rate, latest_new_cases]
])[0]

# ==============================
# Risk Label
# ==============================
def get_risk_label(pred):
    if pred == 0:
        return "🟢 Low Risk"
    elif pred == 1:
        return "🟠 Medium Risk"
    else:
        return "🔴 High Risk"

risk_label = get_risk_label(prediction)

# ==============================
# Header
# ==============================
st.title("🌍 EpiWatch - Epidemic Intelligence Dashboard")

# ==============================
# KPIs
# ==============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Cases", f"{int(latest_cases):,}")
col2.metric("Total Deaths", f"{int(latest_deaths):,}")
col3.metric("New Cases", f"{int(latest_new_cases):,}")
col4.metric("Risk Level", risk_label)

# ==============================
# Infection Trend
# ==============================
fig1 = px.line(df, x="date", y="total_cases", title="Total Cases Over Time")
st.plotly_chart(fig1, use_container_width=True)

# ==============================
# Daily Cases
# ==============================
fig2 = px.bar(df, x="date", y="new_cases", title="Daily New Cases")
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Forecast (Prophet)
# ==============================
st.subheader("📈 Forecast (Next 60 Days)")

forecast_df = df[["date", "total_cases"]].dropna()
forecast_df = forecast_df.rename(columns={"date": "ds", "total_cases": "y"})

if len(forecast_df) > 50:
    prophet_model = Prophet()
    prophet_model.fit(forecast_df)

    future = prophet_model.make_future_dataframe(periods=60)
    forecast = prophet_model.predict(future)

    fig_fc = go.Figure()

    fig_fc.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["y"],
        name="Actual",
        line=dict(color="green")
    ))

    fig_fc.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        name="Forecast",
        line=dict(color="red", dash="dot")
    ))

    st.plotly_chart(fig_fc, use_container_width=True)

# ==============================
# 🌍 Heatmap
# ==============================
st.subheader("🌍 Global Infection Heatmap")

latest = data.sort_values("date").groupby("location").tail(1)

fig_map = px.choropleth(
    latest,
    locations="iso_code",
    color="total_cases",
    hover_name="location",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig_map, use_container_width=True)