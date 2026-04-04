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
├── .streamlit/        # Streamlit configuration
├── dashboard/
│   └── app.py         # Main entry point for the dashboard
├── data/
│   └── owid-covid-data.csv # COVID-19 dataset
├── .gitignore         # Git exclusion rules
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
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

---
*Developed for Global Health Intelligence & Predictive Analytics.*
