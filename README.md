# 🏦 Loan Approval System

A Machine Learning project that predicts whether a loan application should be **Approved** or **Rejected** based on financial indicators and credit risk factors.

This system helps automate loan decision-making using data-driven analysis and reduces the risk of loan defaults.

---

# 📌 Problem Statement

Financial institutions process thousands of loan applications daily.
Manual evaluation of loan eligibility is:

* Time-consuming
* Prone to human bias
* Risky in terms of financial defaults

Banks must evaluate key financial factors such as income, loan amount, tenure, and credit score to assess repayment capacity.

This project builds an intelligent Machine Learning model to automate loan approval decisions and improve risk assessment accuracy.

---

# 🎯 Project Objectives

* Build a credit risk prediction model
* Engineer financial risk-based features
* Compare multiple ML models
* Select the best performing model
* Deploy the model using Streamlit
* Provide real-time prediction with confidence score

---

# 📊 Features Used for Model Training

The model is trained using the following engineered and financial features:

* `loan_income_ratio` → Loan Amount / Annual Income
* `emi_estimate` → Estimated EMI
* `emi_estimate_ratio` → EMI / Annual Income
* `risk_category` → Credit risk category derived from CIBIL score
* `cibil_score` → Credit score (300–900)
* `loan_term` → Loan duration in months

---

# 🧠 Risk Category Logic

| CIBIL Score Range | Risk Category      |
| ----------------- | ------------------ |
| ≤ 550             | 0 → Very High Risk |
| 551–650           | 1 → High Risk      |
| 651–750           | 2 → Medium Risk    |
| > 750             | 3 → Low Risk       |

---

# 🤖 Machine Learning Models Compared

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

Models were compared using:

* Accuracy
* Precision
* Recall
* F1-Score

The best performing model was selected and saved as:

```
best_loan_model.pkl
```

---

# 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

---

# 📂 Project Structure

```
Loan_Approval_Project/
│
├── loan_approval_dataset.csv
├── Loan_Approval_Notebook.ipynb
├── loan_model_training.py
├── best_loan_model.pkl
├── app.py
├── requirement.txt
├── Problem_Statement.txt
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```
git clone https://github.com/VivekSonawane7/Loan-Approval.git
cd loan-approval-ml
```

## 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install pandas numpy scikit-learn streamlit joblib
```

---

# 🚀 Train the Model

Run:

```
python loan_model_training.py
```

This will:

* Train multiple models
* Compare performance
* Save the best model

---

# 🌐 Run Streamlit App

```
streamlit run app.py
```

The web app will open in your browser.

---

# 🖥️ Application Interface

The Streamlit app allows users to:

* Enter Annual Income
* Enter Loan Amount
* Enter Loan Term
* Enter CIBIL Score
* Get Loan Approval Prediction
* View Confidence Level
* See Risk Category Explanation

---

# 📈 Model Evaluation Metrics

The best model was selected based on F1-Score to ensure balanced performance between precision and recall.

Why F1-score?

* Accuracy alone can be misleading in financial datasets
* F1-score balances false approvals and false rejections

---

# 💡 Key Learnings

* Feature engineering significantly improves model performance
* Financial ratios are stronger predictors than raw values
* Ensemble models outperform simple linear models
* Proper feature consistency is critical for deployment
* Deployment requires exact feature alignment with training

---

# 🔮 Future Improvements

* Add ROC-AUC comparison
* Implement Hyperparameter Tuning
* Add SHAP Explainability
* Deploy on AWS / Render / Streamlit Cloud
* Add PDF Loan Decision Report generation
* Integrate real banking API simulation

---

# 🏆 Resume Description

Developed a Machine Learning-based loan approval prediction system using financial risk indicators. Compared multiple models and deployed the best-performing model using Streamlit for real-time credit risk assessment.

---

# 👨‍💻 Author

Vivek Pradip Sonawane
B.Tech Computer Engineering
Aspiring Data Analyst / Machine Learning Engineer

---
