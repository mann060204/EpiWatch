# EpiWatch: Full Technical Documentation & Project Analysis

## 1. Project Background & Vision
EpiWatch is an advanced **Epidemic Intelligence Platform** designed for real-time global health surveillance. The project’s mission is to bridge the gap between raw epidemiological data and actionable public health directives using a multi-layered analytical approach.

By integrating traditional mathematical modeling with modern machine learning, EpiWatch provides health officials with a 360-degree view of viral transmission, mortality trends, and predictive risks.

---

## 2. System Architecture
The application is built on a **Streamlit-driven micro-engine** architecture:
-   **Data Layer**: Ingests high-resolution global COVID-19 data from *Our World in Data (OWID)*.
-   **Intelligence Engine**: An autonomous backend that benchmarks multiple ML models on every session to ensure the highest predictive accuracy.
-   **Simulations Layer**: A mathematical compartment modeling system (SEIR) that solves ordinary differential equations (ODEs) to simulate viral trajectories.
-   **Frontend**: A high-performance "Apple Sleek" clinical dashboard with interactive Plotly visualizations.

---

## 3. Core Algorithms & Machine Learning

### 3.1 Autonomous 4-Layer Benchmarking
EpiWatch does not rely on a single model. Upon initialization, the system benchmarks four distinct architectures:
1.  **Random Forest Classifier**: Best for capturing non-linear relationships in tabular health data.
2.  **Gradient Boosting (GBM)**: Optimized for minimizing predictive errors through sequential tree building.
3.  **XGBoost**: Extreme Gradient Boosting used for high-speed, high-accuracy classification.
4.  **Stacking Ensemble**: Combined architecture using RF and XGBoost as base learners and a GBM as a meta-classifier.

**Model Selection Logic**: The system automatically selects the "Active Model" based on the highest test-set accuracy and lowest **False Omission Rate (FOR)**.

### 3.2 Predictive Feature Engineering
Data is transformed into three primary features for the classifier:
-   `total_cases`: Cumulative volume of infection.
-   `total_deaths`: Cumulative mortality count.
-   `population`: Normalization factor for scaling risks relative to country size.

### 3.3 Explainability (SHAP & Importance)
The system utilizes feature importance scores (XGBoost) and SHAP-inspired analysis to quantify which factors (e.g., mortality rate vs. new case volume) are driving the current risk assessment.

---

## 4. Time-Series Forecasting
### Facebook Prophet Integration
Prophet is used to generate **60-day infection forecasts**. Unlike standard regressions, Prophet handles:
-   **Seasonality**: Captures weekly and yearly patterns in reporting.
-   **Trend Shifts**: Adjusts for rapid changes in transmission (e.g., new variants).
-   **Uncertainty Intervals**: Provides upper and lower confidence bands (`yhat_upper`/`yhat_lower`) to account for statistical variance.

---

## 5. Mathematical SEIR Modeling
For theoretical modeling, EpiWatch uses the **SEIR (Susceptible-Exposed-Infectious-Recovered)** compartmental model:
-   **S (Susceptible)**: Individuals who can catch the virus.
-   **E (Exposed)**: Individuals in the incubation period.
-   **I (Infectious)**: Individuals actively spreading the virus.
-   **R (Recovered/Removed)**: Individuals no longer susceptible.

**The ODE System**:
The system uses `scipy.integrate.odeint` to solve:
-   $dS/dt = -β \cdot S \cdot I / N$
-   $dE/dt = β \cdot S \cdot I / N - σ \cdot E$
-   $dI/dt = σ \cdot E - γ \cdot I$
-   $dR/dt = γ \cdot I$

**Parameters**:
-   **R0 (Basic Reproduction Number)**: Controlled by the user via the UI to simulate the impact of social distancing.
-   **Incubation/Infectious Periods**: Adjust the "velocity" of the theoretical outbreak.

---

## 6. Dashboard Components & UI
### 6.1 Clinical KPI Cards
The dashboard features an interactive grid of KPI cards:
-   **Total Cases/Deaths**: Snapshot of current magnitude.
-   **Mortality Rate**: Immediate calculation of case-fatality ratio.
-   **Historical Peak Risk**: Reminds users of the worst-case scenario recorded for the selected country.
-   **Current ML Risk**: The real-time prediction from the benchmarked ensemble.

### 6.2 Contextual Health Directives
A dynamic recommendation engine generates instructions based on three logical pillars:
1.  **Surveillance**: Comparing current risk vs. historical peak.
2.  **Medical Preparedness**: Triggered by mortality rate spikes.
3.  **Public Guidance**: Based on specific case-volume thresholds.

---

## 7. Technical Implementation Details
-   **Caching**: Uses `@st.cache_data` for data loading and `@st.cache_resource` for model training to optimize performance.
-   **Styling**: Custom CSS injection for a clean, unified aesthetic utilizing modern sans-serif typography (`Inter`, `Apple-system`).
-   **Responsive Layout**: Wide-mode optimization with multi-column distribution for complex data overlays.

---
**EpiWatch Intelligence Platform**
*Documentation Version 1.0*
