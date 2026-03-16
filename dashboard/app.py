import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="EpiWatch · Epidemic Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# Custom CSS — Dark Clinical Theme
# ==============================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

/* ── Root Variables ─────────────────────────────── */
:root {
    --bg-primary:    #060d1a;
    --bg-card:       #0b1628;
    --bg-card-hover: #0f1e38;
    --border:        rgba(0,210,200,0.12);
    --border-glow:   rgba(0,210,200,0.35);
    --accent-teal:   #00d2c8;
    --accent-red:    #ff4554;
    --accent-amber:  #f5a623;
    --accent-purple: #9b59f5;
    --text-primary:  #e8edf5;
    --text-secondary:#7a8ba8;
    --text-dim:      #3d5070;
    --font-display:  'Syne', sans-serif;
    --font-mono:     'Space Mono', monospace;
    --font-body:     'Inter', sans-serif;
}

/* ── Global Reset ───────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

.stApp {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,210,200,0.06) 0%, transparent 60%),
                var(--bg-primary) !important;
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
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: var(--accent-teal);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.dashboard-title {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1;
}
.dashboard-title span {
    color: var(--accent-teal);
}
.header-right {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.live-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--accent-teal);
    border: 1px solid var(--border-glow);
    border-radius: 2rem;
    padding: 0.35rem 0.85rem;
    background: rgba(0,210,200,0.05);
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-teal);
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity:1; transform:scale(1); }
    50%       { opacity:0.4; transform:scale(0.7); }
}

/* ── Section Labels ─────────────────────────────── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}
.section-title {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 1em;
    background: var(--accent-teal);
    border-radius: 2px;
}

/* ── KPI Cards ──────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, background 0.25s;
}
.kpi-card:hover {
    border-color: var(--border-glow);
    background: var(--bg-card-hover);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.teal::before  { background: linear-gradient(90deg, transparent, var(--accent-teal), transparent); }
.kpi-card.red::before   { background: linear-gradient(90deg, transparent, var(--accent-red), transparent); }
.kpi-card.amber::before { background: linear-gradient(90deg, transparent, var(--accent-amber), transparent); }
.kpi-card.purple::before{ background: linear-gradient(90deg, transparent, var(--accent-purple), transparent); }

.kpi-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-family: var(--font-display);
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.kpi-card.teal  .kpi-value { color: var(--accent-teal); }
.kpi-card.red   .kpi-value { color: var(--accent-red); }
.kpi-card.amber .kpi-value { color: var(--accent-amber); }
.kpi-card.purple .kpi-value{ color: var(--accent-purple); }

.kpi-sub {
    font-size: 0.75rem;
    color: var(--text-dim);
}

/* ── Risk Badge ─────────────────────────────────── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.45rem 1rem;
    border-radius: 4px;
}
.risk-low    { background: rgba(0,210,200,0.12); color: var(--accent-teal);   border: 1px solid rgba(0,210,200,0.3); }
.risk-medium { background: rgba(245,166,35,0.12); color: var(--accent-amber);  border: 1px solid rgba(245,166,35,0.3); }
.risk-high   { background: rgba(255,69,84,0.12);  color: var(--accent-red);    border: 1px solid rgba(255,69,84,0.3); }

/* ── Chart Panels ───────────────────────────────── */
.chart-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.chart-panel:hover {
    border-color: var(--border-glow);
}

/* ── Divider ────────────────────────────────────── */
.custom-divider {
    border: none;
    height: 1px;
    background: var(--border);
    margin: 2rem 0;
}

/* ── Table ──────────────────────────────────────── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.78rem;
}
.styled-table th {
    text-align: left;
    padding: 0.6rem 1rem;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
}
.styled-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--text-primary);
}
.styled-table tr:hover td {
    background: rgba(0,210,200,0.03);
}
.rank-num {
    color: var(--text-dim);
    margin-right: 0.75rem;
}
.cases-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    height: 4px;
    width: 100%;
    margin-top: 3px;
}
.cases-bar {
    background: var(--accent-teal);
    height: 4px;
    border-radius: 2px;
}

/* ── Country Selector ───────────────────────────── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
.stSelectbox > div > div:hover {
    border-color: var(--border-glow) !important;
}
label[data-testid="stWidgetLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

/* ── Streamlit overrides ────────────────────────── */
.stPlotlyChart > div { background: transparent !important; }
footer, #MainMenu, header { display: none !important; }
.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1400px !important; }
div[data-testid="stMetric"] { display: none !important; }
div[data-testid="metric-container"] { display: none !important; }
hr { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Dataset
# ==============================

data = pd.read_csv("data/owid-covid-data.csv")
data["date"] = pd.to_datetime(data["date"])

# ==============================
# Train ML Model
# ==============================

ml_data = data[["total_cases", "total_deaths", "population", "new_cases"]].dropna()
ml_data = ml_data.copy()
ml_data["severity"] = 0
ml_data.loc[ml_data["new_cases"] > 1000, "severity"] = 1
ml_data.loc[ml_data["new_cases"] > 10000, "severity"] = 2

X = ml_data[["total_cases", "total_deaths", "population"]]
y = ml_data["severity"]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# ==============================
# Plotly Theme Helper
# ==============================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#7a8ba8", size=11),
    title_font=dict(family="Syne, sans-serif", color="#e8edf5", size=15),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
        zeroline=False, linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=10, color="#3d5070")
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
        zeroline=False, linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=10, color="#3d5070")
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.08)",
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
# Country Selector
# ==============================

country = st.selectbox(
    "Select Country / Region",
    sorted(data["location"].dropna().unique()),
    index=sorted(data["location"].dropna().unique()).index("United States")
        if "United States" in data["location"].values else 0
)

df = data[data["location"] == country]

# ==============================
# Compute Metrics
# ==============================

latest_cases  = df["total_cases"].dropna().iloc[-1]  if not df["total_cases"].dropna().empty  else 0
latest_deaths = df["total_deaths"].dropna().iloc[-1] if not df["total_deaths"].dropna().empty else 0
population    = df["population"].dropna().iloc[-1]   if not df["population"].dropna().empty   else 0
mortality_rate = latest_deaths / latest_cases if latest_cases > 0 else 0

prediction = rf_model.predict([[latest_cases, latest_deaths, population]])[0]
severity_map   = {0: ("Low Risk",    "risk-low",    "○"),
                  1: ("Medium Risk", "risk-medium", "◑"),
                  2: ("High Risk",   "risk-high",   "●")}
risk_label, risk_class, risk_icon = severity_map[prediction]

# ==============================
# KPI Cards
# ==============================

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card teal">
    <div class="kpi-label">Total Confirmed Cases</div>
    <div class="kpi-value">{int(latest_cases):,}</div>
    <div class="kpi-sub">Cumulative count</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Total Deaths</div>
    <div class="kpi-value">{int(latest_deaths):,}</div>
    <div class="kpi-sub">Cumulative fatalities</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">Mortality Rate</div>
    <div class="kpi-value">{mortality_rate*100:.2f}%</div>
    <div class="kpi-sub">Deaths / confirmed cases</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">ML Risk Assessment</div>
    <div class="kpi-value" style="font-size:1.3rem; padding-top:0.2rem;">
      <span class="risk-badge {risk_class}">{risk_icon} {risk_label}</span>
    </div>
    <div class="kpi-sub">Random Forest prediction</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Infection Growth + Daily Cases (2-col)
# ==============================

st.markdown('<div class="section-title">Case Progression</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["total_cases"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,210,200,0.06)",
        line=dict(color="#00d2c8", width=2),
        name="Total Cases",
        hovertemplate="%{y:,.0f}<extra></extra>"
    ))
    fig1.update_layout(title="Cumulative Confirmed Cases", **PLOT_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["date"], y=df["new_cases"],
        marker_color="rgba(0,210,200,0.35)",
        marker_line_width=0,
        name="Daily Cases",
        hovertemplate="%{y:,.0f}<extra></extra>"
    ))
    fig2.update_layout(title="Daily New Infections", bargap=0.1, **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Death Trend
# ==============================

st.markdown('<div class="section-title">Mortality Tracking</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df["date"], y=df["total_deaths"],
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(255,69,84,0.06)",
    line=dict(color="#ff4554", width=2),
    name="Total Deaths",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig3.update_layout(title="Cumulative Death Toll", **PLOT_LAYOUT)
st.plotly_chart(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Forecast
# ==============================

st.markdown('<div class="section-title">Predictive Modelling · Prophet Forecast (60 Days)</div>',
            unsafe_allow_html=True)
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)

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
    fillcolor="rgba(255,69,84,0.07)",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))
fig_fc.add_trace(go.Scatter(
    x=forecast_df["ds"], y=forecast_df["y"],
    mode="lines",
    line=dict(color="#00d2c8", width=2.5),
    name="Actual",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_fc.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode="lines",
    line=dict(color="#ff4554", width=2, dash="dot"),
    name="Forecast",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_fc.update_layout(title="Actual vs. Forecast — Total Cases", **PLOT_LAYOUT)
st.plotly_chart(fig_fc, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Wave Detection
# ==============================

st.markdown('<div class="section-title">Epidemic Wave Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)

df_wave = df.copy()
df_wave["7_day_avg"] = df_wave["new_cases"].rolling(7).mean()

fig_wave = go.Figure()
fig_wave.add_trace(go.Scatter(
    x=df_wave["date"], y=df_wave["new_cases"],
    mode="lines",
    line=dict(color="rgba(255,255,255,0.08)", width=1),
    name="Daily Cases",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_wave.add_trace(go.Scatter(
    x=df_wave["date"], y=df_wave["7_day_avg"],
    mode="lines",
    line=dict(color="#f5a623", width=2.5),
    name="7-Day Average",
    hovertemplate="%{y:,.0f}<extra></extra>"
))
fig_wave.update_layout(title="Daily Cases vs. 7-Day Rolling Average", **PLOT_LAYOUT)
st.plotly_chart(fig_wave, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Global Heatmap
# ==============================

st.markdown('<div class="section-title">Global Distribution</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)

latest = data.sort_values("date").groupby("location").tail(1)
fig_map = px.choropleth(
    latest,
    locations="iso_code",
    color="total_cases",
    hover_name="location",
    color_continuous_scale=[[0,"#0b1628"],[0.3,"#004d5e"],[0.7,"#00a89a"],[1,"#00d2c8"]],
    title="Global COVID-19 Case Distribution"
)
fig_map.update_layout(
    **PLOT_LAYOUT,
    geo=dict(
        bgcolor="rgba(0,0,0,0)",
        landcolor="#0f1e38",
        oceancolor="#060d1a",
        showocean=True,
        lakecolor="#060d1a",
        showland=True,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.06)"
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
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Top 10 + Fastest Growing (2-col)
# ==============================

st.markdown('<div class="section-title">Country Rankings</div>', unsafe_allow_html=True)

col_c, col_d = st.columns([1, 1.4])

with col_c:
    st.markdown('<div class="chart-panel" style="height:100%">', unsafe_allow_html=True)
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
          <td style="text-align:right; color:var(--accent-teal)">{int(row['total_cases']):,}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Country</th><th style="text-align:right">Cases</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_d:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
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
            colorscale=[[0,"#003d3b"],[0.5,"#007a72"],[1,"#00d2c8"]],
            showscale=False,
            line=dict(width=0)
        ),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>"
    ))
    growth_layout = {**PLOT_LAYOUT}
    growth_layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "autorange": "reversed"}
    fig_growth.update_layout(title="Peak Daily New Cases by Country", **growth_layout)
    st.plotly_chart(fig_growth, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Footer
# ==============================

st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1rem; 
     font-family: var(--font-mono); font-size: 0.6rem; 
     letter-spacing:0.15em; color: var(--text-dim);">
  EPIWATCH INTELLIGENCE PLATFORM &nbsp;·&nbsp; DATA: OUR WORLD IN DATA &nbsp;·&nbsp; MODEL: PROPHET + RANDOM FOREST
</div>
""", unsafe_allow_html=True)