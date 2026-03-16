import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("data/owid-covid-data.csv")

# Select important features
data = data[[
    "total_cases",
    "new_cases",
    "total_deaths",
    "population"
]].dropna()

# Create severity label
data["severity"] = 0

data.loc[data["new_cases"] > 1000, "severity"] = 1
data.loc[data["new_cases"] > 10000, "severity"] = 2

# Features and labels
X = data[[
    "total_cases",
    "total_deaths",
    "population"
]]

y = data["severity"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))