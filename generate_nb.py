import nbformat as nbf

nb = nbf.v4.new_notebook()

code_cells = [
    """\
# Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize

plt.style.use('dark_background')""",
    
    """\
# Load Data
data = pd.read_csv('../data/owid-covid-data.csv')
ml_data = data[['total_cases', 'total_deaths', 'population', 'new_cases']].dropna()

# Sample the data to 50,000 rows for faster evaluation
ml_data = ml_data.sample(n=50000, random_state=42)

# Create target variable based on threshold
ml_data['severity'] = 0
ml_data.loc[ml_data['new_cases'] > 1000, 'severity'] = 1
ml_data.loc[ml_data['new_cases'] > 10000, 'severity'] = 2

X = ml_data[['total_cases', 'total_deaths', 'population']]
y = ml_data['severity']

# It is a large dataset, let's take a sample for speedy evaluation (optional)
# But we will use the full set or a solid test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")""",

    """\
# Define Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Stacking Ensemble': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ],
        final_estimator=GradientBoostingClassifier(random_state=42),
        n_jobs=-1
    )
}

results = {}
predictions = {}
predict_probas = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    predictions[name] = y_pred
    predict_probas[name] = y_prob
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1
    print(f"{name} Weighted F1-Score: {f1:.4f}")""",

    """\
# Compare Performance
best_model_name = max(results, key=results.get)
print(f"\\n--- BEST MODEL: {best_model_name} (F1: {results[best_model_name]:.4f}) ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, name in enumerate(models.keys()):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f"Confusion Matrix: {name}")
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()""",

    """\
# ROC AUC Curves for the Models
# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, name in enumerate(models.keys()):
    y_prob = predict_probas[name]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        axes[idx].plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')
    
    axes[idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[idx].set_xlim([0.0, 1.0])
    axes[idx].set_ylim([0.0, 1.05])
    axes[idx].set_xlabel('False Positive Rate')
    axes[idx].set_ylabel('True Positive Rate')
    axes[idx].set_title(f'ROC Curve: {name}')
    axes[idx].legend(loc="lower right")

plt.tight_layout()
plt.show()"""
]

cells = []
for code in code_cells:
    cells.append(nbf.v4.new_code_cell(code))

nb['cells'] = cells

with open('notebooks/model_evaluation.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated at notebooks/model_evaluation.ipynb")
