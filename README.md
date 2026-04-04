# EpiWatch · Epidemic Intelligence Dashboard

EpiWatch is a real-time global health surveillance system designed to monitor, analyze, and forecast epidemic trends using advanced machine learning models.

## 🚀 Key Features

-   **Global Surveillance**: Interactive map visualization of COVID-19 case distributions worldwide.
-   **Infection Growth Monitoring**: Real-time tracking of cumulative and daily new infections.
-   **ML Risk Assessment**: Automated risk categorization (Low, Medium, High) using a **Random Forest Classifier**.
-   **Predictive Modelling**: 60-day future forecasting of infection trends powered by **Facebook Prophet**.
-   **Wave Detection**: Visualization of daily cases vs. 7-day rolling averages for epidemic wave identification.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit (Clinical Dark Theme)
-   **Data Processing**: Pandas, Scikit-learn
-   **Visualizations**: Plotly (Interactive Charts & Choropleth Maps)
-   **Forecasting**: Facebook Prophet
-   **Machine Learning**: Random Forest Classifier

## 📋 Project Structure
```text
Epidemic_Project/
├── dashboard/
│   ├── app.py         # Main entry point (Latest Engine)
│   └── app1.py        # Development Version
├── src/               # Modular Analysis Scripts
│   ├── data_cleaning.py
│   ├── features.py
│   ├── growth_model.py
│   ├── load_data.py
│   ├── severity_model.py
│   └── severity_ranking.py
├── notebooks/         # Exploratory Analysis
│   └── model_evaluation.ipynb
├── data/
│   └── owid-covid-data.csv # Data Source
├── .streamlit/        # Dashboard Config
├── generate_nb.py     # Notebook Generator
├── README.md          # Project Overview
├── DOCUMENTATION.md   # Technical Reference
└── requirements.txt   # Dependencies
```

## 📋 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃 Running the Dashboard

Once the dependencies are installed, start the application by running:

```bash
streamlit run dashboard/app.py
```

## 📊 Data Source
This project uses data provided by **Our World in Data (OWID)**.

```
Dataset Link: https://www.kaggle.com/datasets/bolkonsky/covid19
```
---
*Developed for Global Health Intelligence & Predictive Analytics.*
