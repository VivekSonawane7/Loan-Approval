# ============================================
# Loan Approval Prediction App
# (Multi-Model Compatible Version)
# ============================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    return joblib.load("best_loan_model.pkl")

model = load_model()

# ============================================
# TITLE
# ============================================

st.title("🏦 Loan Approval Prediction System")
st.markdown("### Enter Applicant Details")

st.markdown("---")

# ============================================
# USER INPUTS
# ============================================

income_annum = st.number_input(
    "Annual Income",
    min_value=1.0,
    value=500000.0,
    help="Total yearly income of the applicant"
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=1.0,
    value=200000.0,
    help="Requested loan amount"
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=60,
    help="Duration of loan in months"
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=750,
    help="Credit score (300-900)"
)

# ============================================
# FEATURE ENGINEERING
# ============================================

def get_risk(score):
    if score <= 550:
        return 0
    elif score <= 650:
        return 1
    elif score <= 750:
        return 2
    else:
        return 3

# ============================================
# PREDICTION
# ============================================

if st.button("🔍 Predict Loan Status"):

    # Create required features
    loan_income_ratio = loan_amount / income_annum
    emi_estimate = loan_amount / loan_term
    emi_estimate_ratio = emi_estimate / income_annum
    risk_category = get_risk(cibil_score)

    input_df = pd.DataFrame([{
        "loan_income_ratio": loan_income_ratio,
        "emi_estimate": emi_estimate,
        "emi_estimate_ratio": emi_estimate_ratio,
        "risk_category": risk_category,
        "cibil_score": cibil_score,
        "loan_term": loan_term
    }])

    proba = model.predict_proba(input_df)[0]
    prediction = np.argmax(proba)
    confidence = float(np.max(proba))

    st.markdown("---")

    # Result Display
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"### Confidence Level: {confidence:.2%}")
    st.progress(confidence)

    # Feature Explanation
    st.markdown("---")
    st.subheader("📊 Feature Summary")

    st.write("Loan to Income Ratio:", round(loan_income_ratio, 4))
    st.write("Estimated EMI:", round(emi_estimate, 2))
    st.write("EMI to Income Ratio:", round(emi_estimate_ratio, 4))
    st.write("Risk Category:", risk_category)

    st.info("""
    Risk Category Meaning:
    0 → Very High Risk
    1 → High Risk
    2 → Medium Risk
    3 → Low Risk
    """)
