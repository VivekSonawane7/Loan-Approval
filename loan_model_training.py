# ============================================
# Loan Approval - Multiple Model Comparison
# (Without XGBoost)
# ============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# ============================================
# LOAD DATA
# ============================================

data = pd.read_csv("loan_approval_dataset.csv")
data.columns = data.columns.str.strip()

data["loan_status"] = (
    data["loan_status"]
    .str.strip()
    .str.lower()
    .map({"approved": 1, "rejected": 0})
)

# ============================================
# FEATURE ENGINEERING
# ============================================

data['loan_income_ratio'] = data['loan_amount'] / data['income_annum']
data['emi_estimate'] = data['loan_amount'] / data['loan_term']
data['emi_estimate_ratio'] = data['emi_estimate'] / data['income_annum']

def get_risk(score):
    if score <= 550:
        return 0
    elif score <= 650:
        return 1
    elif score <= 750:
        return 2
    else:
        return 3

data["risk_category"] = data["cibil_score"].apply(get_risk)

# ============================================
# SELECT ONLY REQUIRED FEATURES
# ============================================

features = [
    "loan_income_ratio",
    "emi_estimate",
    "emi_estimate_ratio",
    "risk_category",
    "cibil_score",
    "loan_term"
]

X = data[features]
y = data["loan_status"]

# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# DEFINE MODELS
# ============================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# ============================================
# TRAIN & EVALUATE
# ============================================

results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append([name, accuracy, precision, recall, f1])

# ============================================
# RESULTS TABLE
# ============================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

results_df = results_df.sort_values(by="F1-Score", ascending=False)

print("\nModel Comparison Results:\n")
print(results_df)

# ============================================
# SAVE BEST MODEL
# ============================================

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

joblib.dump(best_model, "best_loan_model.pkl")

print(f"\nBest Model Saved: {best_model_name}")