import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.integrate import odeint
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="EpiWatch · Epidemic Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Sidebar - Analysis Settings
# ==============================
with st.sidebar:
    st.markdown('<div class="dashboard-eyebrow">Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: var(--accent-blue); margin-bottom: 1rem;">✓ Autonomous Model Benchmarking Active</div>', unsafe_allow_html=True)
    
    def render_recommendation(title, desc):
        st.markdown(f"""
        <div style="border-left: 3px solid var(--accent-blue); padding-left: 1.2rem; margin-bottom: 1rem; background: rgba(0,113,227,0.02); padding: 1rem; border-radius: 8px;">
            <div class="section-label" style="color: var(--accent-blue); margin-bottom: 0.4rem;">{title}</div>
            <div style="font-size: 0.82rem; line-height: 1.5; color: var(--text-primary); font-weight: 500;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    show_shap = st.checkbox("Show Model Interpretability (SHAP)", value=True)
    show_metrics = st.checkbox("Show Detailed Evaluation", value=True)
    
    # Time Horizon is initialized later after data loading

# ==============================
# Custom CSS — Apple Sleek Theme
# ==============================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root Variables - Apple Sleek Theme ── */
:root {
    --bg-primary:    #fbfbfd;
    --bg-card:       #ffffff;
    --bg-card-hover: #ffffff;
    --border:        rgba(0,0,0,0.06);
    --border-glow:   rgba(0,113,227,0.3);
    --accent-teal:   #32ade6;
    --accent-red:    #ff3b30;
    --accent-amber:  #ff9500;
    --accent-purple: #af52de;
    --accent-blue:   #0071e3;
    --text-primary:  #1d1d1f;
    --text-secondary:#86868b;
    --text-dim:      #a1a1a6;
    --shadow-subtle: 0 4px 20px rgba(0,0,0,0.04);
    --shadow-hover:  0 8px 30px rgba(0,0,0,0.08);
    --font-body:     'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Global Reset ───────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

.stApp {
    background: var(--bg-primary) !important;
}

/* ── Header Banner ──────────────────────────────── */
.dashboard-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.header-left {}
.dashboard-eyebrow {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.dashboard-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.04em;
    line-height: 1;
}
.dashboard-title span {
    color: var(--accent-blue);
}
.header-right {
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* ── Section Labels ─────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.02em;
}

/* ── KPI Cards ──────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    position: relative;
    box-shadow: var(--shadow-subtle);
    transition: all 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}

.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.4rem;
    color: var(--text-primary);
}
.kpi-card.blue  .kpi-value { color: var(--accent-blue); }
.kpi-card.red   .kpi-value { color: var(--accent-red); }
.kpi-card.amber .kpi-value { color: var(--accent-amber); }
.kpi-card.purple .kpi-value{ color: var(--accent-purple); }

.kpi-sub {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* ── Risk Badge ─────────────────────────────────── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.4rem 1rem;
    border-radius: 8px;
}
.risk-low    { background: rgba(50, 173, 230, 0.1); color: var(--accent-blue); }
.risk-medium { background: rgba(255, 149, 0, 0.1); color: var(--accent-amber); }
.risk-high   { background: rgba(255, 59, 48, 0.1); color: var(--accent-red); }

/* ── Chart Panels ───────────────────────────────── */
div[data-testid="stPlotlyChart"], div[data-testid="stTable"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: var(--shadow-subtle) !important;
    transition: box-shadow 0.3s ease !important;
}
div[data-testid="stPlotlyChart"]:hover {
    box-shadow: var(--shadow-hover) !important;
}

/* ── Table ──────────────────────────────────────── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.styled-table th {
    text-align: left;
    padding: 0.8rem 1rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
}
.styled-table td {
    padding: 0.8rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
    font-weight: 500;
}
.styled-table tr:hover td {
    background: rgba(0,0,0,0.01);
}
.rank-num {
    color: var(--text-dim);
    margin-right: 0.75rem;
    font-weight: 400;
}
.cases-bar-bg {
    background: rgba(0,0,0,0.04);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 6px;
}
.cases-bar {
    background: var(--accent-blue);
    height: 6px;
    border-radius: 4px;
}

/* ── Inputs/Sidebar Overrides ───────────────────────────── */
.stSelectbox > div > div, .stDateInput > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
}
.stSelectbox > div > div:hover, .stDateInput > div > div:hover {
    border-color: rgba(0,0,0,0.15) !important;
}
/* Target text inputs directly so typing isn't white-on-white */
input, input[type="text"], .stSelectbox input, .stDateInput input {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}
/* Target the popup dropdown menus (BaseWeb overlays) */
div[data-baseweb="popover"] > div, 
div[data-baseweb="popover"],
ul[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"] {
    background-color: #ffffff !important;
}
li[role="option"], div[role="option"], ul[data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #1d1d1f !important;
}
li[role="option"]:hover, div[role="option"]:hover, ul[data-baseweb="menu"] li:hover {
    background-color: rgba(0,0,0,0.04) !important;
}
label[data-testid="stWidgetLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* ── Streamlit overrides ────────────────────────── */
.stPlotlyChart > div { background: transparent !important; }
footer, #MainMenu, header { display: none !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1400px !important; }
div[data-testid="stMetric"] { display: none !important; }
div[data-testid="metric-container"] { display: none !important; }
hr { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Dataset
# ==============================

@st.cache_data
def load_owid_data():
    df = pd.read_csv("data/owid-covid-data.csv")
    df["date"] = pd.to_datetime(df["date"])
    # Filter out aggregated regions (like 'World', 'Asia', 'Income groups')
    df = df[df["continent"].notna()]
    return df

data = load_owid_data()


# ==============================
# Core Intelligence Engine (ML)
# ==============================

@st.cache_resource(show_spinner=False)
def train_and_evaluate_all_models(df, n_estimators=50):
    ml_data = df[["total_cases", "total_deaths", "population", "new_cases"]].dropna()
    ml_data = ml_data.copy()
    ml_data["severity"] = 0
    ml_data.loc[ml_data["new_cases"] > 1000, "severity"] = 1
    ml_data.loc[ml_data["new_cases"] > 10000, "severity"] = 2
    
    features = ["total_cases", "total_deaths", "population"]
    X = ml_data[features]
    y = ml_data["severity"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        "Stacking Ensemble": StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=30, random_state=42, n_jobs=-1))
            ],
            final_estimator=GradientBoostingClassifier(n_estimators=20, random_state=42)
        )
    }
    
    results = []
    best_acc = 0
    best_model = None
    best_cm = None
    best_name = ""
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        for_rate = 1 - (cm.diagonal().sum() / cm.sum())
        results.append({"Model": name, "Accuracy (%)": acc * 100.0, "FOR (%)": for_rate * 100.0})
        
        if acc > best_acc:
            best_acc = acc
            best_model = m
            best_cm = cm
            best_name = name
            
    res_df = pd.DataFrame(results).sort_values("Accuracy (%)", ascending=False)
    best_for = 1 - (best_cm.diagonal().sum() / best_cm.sum())
    
    return best_model, best_name, res_df, best_cm, best_for, X_test, features

# Engage robust benchmarking automatically
with st.spinner("Initializing Advanced Autonomous Intelligence Benchmarking (Training 4 Models)..."):
    active_model, active_model_name, metrics_df, conf_matrix, for_score, X_val, feature_names = train_and_evaluate_all_models(data, 50)

# ==============================
# Plotly Theme Helper
# ==============================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#86868b", size=12),
    title_font=dict(color="#1d1d1f", size=16, family="Inter, -apple-system, sans-serif"),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.04)",
        zeroline=False, linecolor="rgba(0,0,0,0.06)",
        tickfont=dict(size=11, color="#86868b")
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.04)",
        zeroline=False, linecolor="rgba(0,0,0,0.06)",
        tickfont=dict(size=11, color="#86868b")
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.06)",
        borderwidth=1,
        font=dict(size=10, color="#7a8ba8")
    ),
    margin=dict(t=50, b=30, l=10, r=10),
    hovermode="x unified"
)

# ==============================
# Header
# ==============================

st.markdown("""
<div class="dashboard-header">
  <div class="header-left">
    <div class="dashboard-eyebrow">Global Health Surveillance System</div>
    <div class="dashboard-title">Epi<span>Watch</span></div>
  </div>
  <div class="header-right">
    <div class="live-badge">
      <div class="live-dot"></div>
      LIVE MONITORING
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Search & Filter
# ==============================

col_search, col_date = st.columns([1, 1], gap="large")

with col_search:
    country = st.selectbox(
        "Select Country / Region",
        sorted(data["location"].dropna().unique()),
        index=sorted(data["location"].dropna().unique()).index("United States")
            if "United States" in data["location"].values else 0
    )

country_data = data[data["location"] == country]

# Handle Dates
min_d = country_data["date"].min().date()
max_d = country_data["date"].max().date()

st.sidebar.markdown('<div class="dashboard-eyebrow" style="margin-top: 1.5rem;">Time Horizon</div>', unsafe_allow_html=True)
date_range = st.sidebar.date_input("Analysis Period", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="daterange")

# We inject the default values manually into the sidebar if not set (due to streamlit logic constraints)
if not date_range or len(date_range) < 2:
    val1 = date_range[0] if date_range else min_d
    val2 = date_range[0] if date_range else max_d
    start_date, end_date = val1, val2
else:
    start_date, end_date = date_range[0], date_range[1]

df = country_data[(country_data["date"].dt.date >= start_date) & (country_data["date"].dt.date <= end_date)]


# ==============================
# Compute Metrics
# ==============================
severity_map   = {0: ("Low Risk",    "risk-low",    "○"),
                  1: ("Medium Risk", "risk-medium", "◑"),
                  2: ("High Risk",   "risk-high",   "●")}

# Historical Peak
peak_cases = country_data["new_cases"].max()
hist_risk_val = 2 if peak_cases > 10000 else (1 if peak_cases > 1000 else 0)
hist_label, hist_class, hist_icon = severity_map[hist_risk_val]

# Current Window End Metrics
latest_cases  = df["total_cases"].dropna().iloc[-1]  if not df["total_cases"].dropna().empty  else 0
latest_deaths = df["total_deaths"].dropna().iloc[-1] if not df["total_deaths"].dropna().empty else 0
population    = df["population"].dropna().iloc[-1]   if not df["population"].dropna().empty   else 0
mortality_rate = latest_deaths / latest_cases if latest_cases > 0 else 0

input_data = pd.DataFrame([[latest_cases, latest_deaths, population]], columns=feature_names)
prediction = active_model.predict(input_data)[0]
risk_label, risk_class, risk_icon = severity_map[prediction]

# ==============================
# KPI Cards (Dual Risk)
# ==============================

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="kpi-label">Total Cases</div>
    <div class="kpi-value">{int(latest_cases):,}</div>
    <div class="kpi-sub">End of selected range</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Total Deaths</div>
    <div class="kpi-value">{int(latest_deaths):,}</div>
    <div class="kpi-sub">End of selected range</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">Mortality Rate</div>
    <div class="kpi-value">{mortality_rate*100:.2f}%</div>
    <div class="kpi-sub">Deaths / confirmed cases</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">Historical Peak Risk</div>
    <div class="kpi-value" style="font-size:1.15rem; padding-top:0.2rem;">
      <span class="risk-badge {hist_class}">{hist_icon} {hist_label}</span>
    </div>
    <div class="kpi-sub">All-time highest severity</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">Current ML Risk</div>
    <div class="kpi-value" style="font-size:1.15rem; padding-top:0.2rem;">
      <span class="risk-badge {risk_class}">{risk_icon} {risk_label}</span>
    </div>
    <div class="kpi-sub">XGBoost prediction</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Infection Growth + Daily Cases (2-col)
# ==============================

st.markdown('<div class="section-title">Case Progression</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["total_cases"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,113,227,0.1)",
        line=dict(color="#0071e3", width=2),
        name="Total Cases",
        hovertemplate="%{y:,.0f}<extra></extra>"
    ))
    fig1.update_layout(title="Cumulative Confirmed Cases", **PLOT_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)
    

with col_b:
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["date"], y=df["new_cases"],
        marker_color="rgba(0,113,227,0.4)",
        marker_line_width=0,
        name="Daily Cases",
        hovertemplate="%{y:,.0f}<extra></extra>"
    ))
    fig2.update_layout(title="Daily New Infections", bargap=0.1, **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)
    

# ==============================
# Death Trend
# ==============================

st.markdown('<div class="section-title">Mortality Tracking</div>', unsafe_allow_html=True)


fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df["date"], y=df["total_deaths"],
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(255,59,48,0.1)",
    line=dict(color="#ff3b30", width=2),
    name="Total Deaths",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig3.update_layout(title="Cumulative Death Toll", **PLOT_LAYOUT)
st.plotly_chart(fig3, use_container_width=True)


# ==============================
# Forecast
# ==============================

st.markdown('<div class="section-title">Predictive Modelling · Prophet Forecast (60 Days)</div>',
            unsafe_allow_html=True)


forecast_df = df[["date", "total_cases"]].dropna().rename(columns={"date": "ds", "total_cases": "y"})

with st.spinner("Running Prophet model…"):
    prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    prophet_model.fit(forecast_df)
    future   = prophet_model.make_future_dataframe(periods=60)
    forecast = prophet_model.predict(future)

fig_fc = go.Figure()
# Confidence band
fig_fc.add_trace(go.Scatter(
    x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(255,59,48,0.08)",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))
fig_fc.add_trace(go.Scatter(
    x=forecast_df["ds"], y=forecast_df["y"],
    mode="lines",
    line=dict(color="#0071e3", width=2.5),
    name="Actual",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_fc.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode="lines",
    line=dict(color="#ff3b30", width=2, dash="dot"),
    name="Forecast",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_fc.update_layout(title="Actual vs. Forecast — Total Cases", **PLOT_LAYOUT)
st.plotly_chart(fig_fc, use_container_width=True)


# ==============================
# Wave Detection
# ==============================

st.markdown('<div class="section-title">Epidemic Wave Detection</div>', unsafe_allow_html=True)


df_wave = df.copy()
df_wave["7_day_avg"] = df_wave["new_cases"].rolling(7).mean()

fig_wave = go.Figure()
fig_wave.add_trace(go.Scatter(
    x=df_wave["date"], y=df_wave["new_cases"],
    mode="lines",
    line=dict(color="rgba(0,0,0,0.15)", width=1),
    name="Daily Cases",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_wave.add_trace(go.Scatter(
    x=df_wave["date"], y=df_wave["7_day_avg"],
    mode="lines",
    line=dict(color="#ff9500", width=2.5),
    name="7-Day Average",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_wave.update_layout(title="Daily Cases vs. 7-Day Rolling Average", **PLOT_LAYOUT)
st.plotly_chart(fig_wave, use_container_width=True)

# ==============================
# Mathematical Epidemiological SEIR Simulation
# ==============================
st.markdown('<div class="section-title">Interactive SEIR Epidemic Modeling (Mathematical Simulation)</div>', unsafe_allow_html=True)
st.markdown("""
<div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1.5rem; max-width: 800px;">
Simulate the theoretical spread of the virus over the next 150 days using a Susceptible-Exposed-Infectious-Recovered (SEIR) mathematical model based on current population sizes. Use the interactive controls to observe how public health interventions flatten the curve.
</div>
""", unsafe_allow_html=True)

col_sim_left, col_sim_right = st.columns([1, 2.5])

with col_sim_left:
    st.markdown('<div class="dashboard-eyebrow" style="margin-bottom: 1rem;">Simulation Parameters</div>', unsafe_allow_html=True)
    r0 = st.slider("Basic Reproduction No. (R0)", 1.1, 5.0, 2.2, 0.1, help="The expected number of secondary cases produced by a single infection. Lowering this simulates lockdowns/masks.")
    t_incubation = st.slider("Incubation Period (Days)", 2, 14, 5, 1, help="Time from exposure to active infectiousness.")
    t_infectious = st.slider("Infectious Period (Days)", 3, 21, 10, 1, help="Duration a patient actively spreads the virus.")
    
    # Compute parameters
    gamma = 1.0 / t_infectious
    beta = r0 * gamma
    sigma = 1.0 / t_incubation

with col_sim_right:
    def seir_deriv(y, t, N, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
        
    N_pop = population if population > 0 else 1000000 
    I0 = latest_cases if latest_cases > 0 else 1 
    E0, R0_state = I0 * 4, 0 
    S0 = N_pop - I0 - E0 - R0_state
    
    y0 = S0, E0, I0, R0_state
    t = np.linspace(0, 150, 150)
    
    ret = odeint(seir_deriv, y0, t, args=(N_pop, beta, sigma, gamma))
    S, E, I, R = ret.T
    
    fig_seir = go.Figure()
    fig_seir.add_trace(go.Scatter(x=t, y=I, mode='lines', line=dict(color="#ff3b30", width=3), name='Infectious', fill="tozeroy", fillcolor="rgba(255,59,48,0.1)"))
    fig_seir.add_trace(go.Scatter(x=t, y=E, mode='lines', line=dict(color="#ff9500", width=2, dash='dot'), name='Exposed'))
    fig_seir.add_trace(go.Scatter(x=t, y=R, mode='lines', line=dict(color="#34c759", width=2), name='Recovered/Removed'))
    
    fig_seir.update_layout(title=f"150-Day Theoretical Trajectory ({country})", xaxis_title="Days", yaxis_title="Population Count", **PLOT_LAYOUT, height=380)
    st.plotly_chart(fig_seir, use_container_width=True)


# ==============================
# Global Heatmap
# ==============================

st.markdown('<div class="section-title">Global Distribution</div>', unsafe_allow_html=True)


latest = data.sort_values("date").groupby("location").tail(1)
fig_map = px.choropleth(
    latest,
    locations="iso_code",
    color="total_cases",
    hover_name="location",
    color_continuous_scale=[[0,"#e5e5ea"],[0.3,"#add8e6"],[0.7,"#0071e3"],[1,"#004080"]],
    title="Global COVID-19 Case Distribution"
)
fig_map.update_layout(
    **PLOT_LAYOUT,
    geo=dict(
        bgcolor="rgba(0,0,0,0)",
        landcolor="#f2f2f7",
        oceancolor="#ffffff",
        showocean=True,
        lakecolor="#ffffff",
        showland=True,
        showcountries=True,
        countrycolor="rgba(0,0,0,0.1)"
    ),
    coloraxis_colorbar=dict(
        title=dict(
            text="Total Cases",
            font=dict(color="#7a8ba8", size=10)
        ),
        tickfont=dict(color="#7a8ba8", size=9)
    )
)
st.plotly_chart(fig_map, use_container_width=True)


# ==============================
# Top 10 + Fastest Growing (2-col)
# ==============================

st.markdown('<div class="section-title">Country Rankings</div>', unsafe_allow_html=True)

col_c, col_d = st.columns([1, 1.4])

with col_c:
    
    st.markdown('<div class="section-label">Most Affected Countries</div>', unsafe_allow_html=True)

    severity_df = (
        data.groupby("location")["total_cases"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    max_cases = severity_df["total_cases"].max()

    rows_html = ""
    for i, row in severity_df.iterrows():
        pct = (row["total_cases"] / max_cases) * 100
        rows_html += f"""
        <tr>
          <td>
            <span class="rank-num">{i+1:02d}</span>
            {row['location']}
            <div class="cases-bar-bg"><div class="cases-bar" style="width:{pct:.1f}%"></div></div>
          </td>
          <td style="text-align:right; color:var(--accent-blue); font-weight: 600;">{int(row['total_cases']):,}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Country</th><th style="text-align:right">Cases</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    

with col_d:
    
    growth = (
        data.groupby("location")["new_cases"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_growth = go.Figure(go.Bar(
        x=growth["new_cases"],
        y=growth["location"],
        orientation="h",
        marker=dict(
            color=growth["new_cases"],
            colorscale=[[0,"#e5e5ea"],[0.5,"#add8e6"],[1,"#0071e3"]],
            showscale=False,
            line=dict(width=0)
        ),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>"
    ))
    growth_layout = {**PLOT_LAYOUT}
    growth_layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "autorange": "reversed"}
    fig_growth.update_layout(title="Peak Daily New Cases by Country", **growth_layout)
    st.plotly_chart(fig_growth, use_container_width=True)
    

# ==============================
# Model Interpretability & Evaluation
# ==============================
if show_shap or show_metrics:
    st.markdown('<div class="section-title">Model Benchmarking & Interpretability</div>', unsafe_allow_html=True)
    if show_metrics:
        # Model Benchmarking Chart
        fig_bench = go.Figure()
        
        # Color highlighting the autonomously selected best model
        colors = ["#0071e3" if m == active_model_name else "#e5e5ea" for m in metrics_df["Model"]]
        
        fig_bench.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["Accuracy (%)"],
            marker_color=colors,
            text=[f"{val:.1f}%" for val in metrics_df["Accuracy (%)"]],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>"
        ))
        
        fig_bench.update_layout(title="Autonomous 4-Layer Intelligence Benchmarking Results", yaxis_title="Accuracy (%)", **PLOT_LAYOUT, height=350)
        st.plotly_chart(fig_bench, use_container_width=True)
        
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        if show_metrics:
            
            st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
            fig_cm = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                               x=['Low', 'Medium', 'High'], y=['Low', 'Medium', 'High'])
            fig_cm.update_layout(**PLOT_LAYOUT, title="Confusion Matrix")
            fig_cm.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # FOR Metric
            st.markdown(f"""
            <div style="background: rgba(255,59,48,0.05); border: 1px solid rgba(255,59,48,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--accent-red); margin-bottom: 0.2rem; text-transform: uppercase;">Global False Omission Rate (FOR)</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #1d1d1f;">{for_score*100:.2f}%</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Proportion of critical misclassifications</div>
            </div>
            """, unsafe_allow_html=True)
            
            
    with col_eval2:
        if show_shap and hasattr(active_model, "feature_importances_"):
            
            st.markdown('<div class="section-label">Predictive Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
            
            importances = active_model.feature_importances_
            fig_imp = go.Figure(go.Bar(
                x=importances,
                y=feature_names,
                orientation='h',
                marker=dict(color="#0071e3")
            ))
            imp_layout = {**PLOT_LAYOUT}
            imp_layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "autorange": "reversed"}
            fig_imp.update_layout(**imp_layout, title="Predictive Feature Importance (XGBoost)")
            fig_imp.update_layout(height=400, margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig_imp, use_container_width=True)
            

# ==============================
# Public Health Recommendations
# ==============================
st.markdown('<div class="section-title">Contextual Health Directives (Driven by ML Risk)</div>', unsafe_allow_html=True)

def get_dynamic_directives(c_name, c_risk, h_risk, cases, mort):
    directives = []
    
    # 1: Surveillance & Monitoring (Historical vs Current context)
    if c_risk == 0:
        if h_risk >= 1:
            d1 = ("Vigilance & Surveillance", f"Though predictive models suggest Low Risk for <strong>{c_name}</strong>, historical data shows severe past outbreaks. Maintain baseline genomic sequencing and wastewater monitoring to prevent hidden flare-ups.")
        else:
            d1 = ("Standard Monitoring", f"<strong>{c_name}</strong> maintains a stable Low Risk profile. Sustain routine epidemiological reporting and promote general seasonal hygiene practices to keep transmission baseline low.")
    elif c_risk == 1:
        d1 = ("Targeted Intervention", f"With Medium Risk actively predicted in <strong>{c_name}</strong>, immediately ramp up localized testing in dense population centers. Deploy rapid response tracking teams to isolate active transmission chains.")
    else:
        d1 = ("Emergency Containment", f"High Risk actively predicted! Enforce strict mobility limits across <strong>{c_name}</strong>. Trigger emergency epidemic protocols to halt exponential spread immediately.")
    directives.append(d1)
    
    # 2: Medical Preparedness (Mortality vs Capacity context)
    if mort > 0.03: 
        d2 = ("Critical Resource Allocation", f"Mortality in <strong>{c_name}</strong> is critically elevated ({mort*100:.1f}%). Prioritize immediate delivery of ventilators, therapeutics, and specialized ICU staff to overburdened medical facilities.")
    elif c_risk == 2:
        d2 = ("Healthcare Mobilization", f"Brace <strong>{c_name}</strong>'s healthcare system for an incoming surge. Activate secondary care reserves and preemptively halt non-essential elective surgeries to free up ICU beds.")
    elif c_risk == 1:
        d2 = ("Resource Auditing", f"Audit regional supply chains in <strong>{c_name}</strong>. Ensure a 60-day baseline stock of PPE, rapid anti-viral therapeutics, and testing kits in all central facilities.")
    else:
        d2 = ("System Recovery", f"Healthcare capacity in <strong>{c_name}</strong> is stable. Use this period to intelligently restock critical medical supplies, evaluate healthcare worker fatigue, and safely fortify infrastructure.")
    directives.append(d2)
    
    # 3: Public Guidance (Scale context)
    if c_risk == 2:
        d3 = ("Mandatory Measures", f"Implement sweeping indoor mask mandates rapidly across <strong>{c_name}</strong>. Strongly advise against mass public gatherings, and mandate work-from-home protocols where economically viable.")
    elif c_risk == 1 and cases > 5000:
        d3 = ("Elevated Advisories", f"With cases rising, strongly recommend high-quality masks in high-risk indoor spaces within <strong>{c_name}</strong>. Issue targeted protective warnings for elderly and immunocompromised demographics.")
    else:
        d3 = ("Public Readiness", f"No immediate sweeping restrictions required for <strong>{c_name}</strong>. Rely on sustained, transparent public health messaging to keep the population informed without causing pandemic fatigue.")
    directives.append(d3)
    
    return directives

current_recs = get_dynamic_directives(country, int(prediction), hist_risk_val, latest_cases, mortality_rate)

rec_cols = st.columns(3)
for i, (title, desc) in enumerate(current_recs):
    with rec_cols[i]:
        # Apply styling explicitly for the new larger, nicer card
        st.markdown(f"""
        <div style="
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-left: 5px solid var(--accent-blue);
            border-radius: 14px;
            padding: 1.8rem;
            height: 100%;
            box-shadow: 0 4px 14px rgba(0,0,0,0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        " onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 8px 24px rgba(0,0,0,0.08)';" onmouseout="this.style.transform='none'; this.style.boxShadow='0 4px 14px rgba(0,0,0,0.03)';">
            <div style="color: var(--accent-blue); font-size: 0.95rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.8rem;">{title}</div>
            <div style="font-size: 0.95rem; line-height: 1.6; color: var(--text-primary); font-weight: 400;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Footer
# ==============================

st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1rem; 
     font-family: var(--font-mono); font-size: 0.6rem; 
     letter-spacing:0.15em; color: var(--text-dim);">
  EPIWATCH INTELLIGENCE PLATFORM &nbsp;·&nbsp; DATA: OUR WORLD IN DATA &nbsp;·&nbsp; MODEL: PROPHET + XGBOOST
</div>
""", unsafe_allow_html=True)