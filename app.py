import streamlit as st


st.set_page_config(page_title="Lanka Micro-Finance AI", page_icon="💰", layout="centered")

import pandas as pd
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "microfinance_model.pkl")
model = None

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "⚠️ Model file not found at `data/microfinance_model.pkl`. "
        "Please train the model and place it in the `data/` folder before running this app."
    )
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")

# --- App Header ---
st.title("💰 Micro-Finance Risk AI")
st.markdown("""
### Alternative Credit Scoring for Small Businesses
This AI system evaluates loan eligibility using behavioral and alternative data points
specifically tailored for the Sri Lankan micro-entrepreneurship context.
""")

st.divider()

# --- Input Form ---
with st.form("risk_assessment_form"):
    st.subheader("📋 Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Applicant Age", min_value=18, max_value=75, value=30)
        income = st.number_input(
            "Monthly Income (LKR)", min_value=5000, max_value=1000000, value=50000, step=5000
        )
        biz_type = st.selectbox(
            "Business Category",
            ["Street Vendor", "Small Retail Shop", "Home-based Business", "Freelancer/Service Provider"],
        )
        late_days = st.number_input(
            "Utility Bill Late Payment Days (Last 30 days)", min_value=0, max_value=30, value=0
        )

    with col2:
        reload = st.slider("Mobile Reload Consistency (0: Poor → 1: Very Consistent)", 0.0, 1.0, 0.8)
        digi_score = st.slider("Digital Literacy Score", 0, 100, 50)
        loans = st.number_input("Existing Active Loans", min_value=0, max_value=10, value=0)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=15, value=1)

    community_member = st.checkbox("Member of a Local Community / Trade Group")

    submitted = st.form_submit_button("🔍 Analyze Credit Risk", use_container_width=True)

# --- Prediction ---
if submitted:
    if model is None:
        st.error("Cannot perform analysis — model is not loaded. See the error above.")
        st.stop()

    biz_map = {
        "Street Vendor": 0,
        "Small Retail Shop": 1,
        "Home-based Business": 2,
        "Freelancer/Service Provider": 3,
    }
    is_member = 1 if community_member else 0

    input_features = pd.DataFrame(
        [[age, income, biz_map[biz_type], late_days, reload, digi_score, is_member, loans, dependents]],
        columns=[
            "Age", "Monthly_Income_LKR", "Business_Type", "Utility_Bill_Late_Days",
            "Mobile_Reload_Consistency", "Digital_Literacy_Score",
            "Community_Group_Member", "Existing_Loans", "Dependents",
        ],
    )

    prediction = model.predict(input_features)[0]
    risk_probability = float(model.predict_proba(input_features)[0][1])

    st.divider()
    st.subheader("📊 Risk Analysis Result")

    if prediction == 0:
        st.success("### ✅ LOW RISK — Recommended for Loan")
        st.write(f"Estimated probability of default: **{risk_probability * 100:.1f}%**")
        st.balloons()
    else:
        st.error("### ⚠️ HIGH RISK — Not Recommended")
        st.write(f"Estimated probability of default: **{risk_probability * 100:.1f}%**")
        factors = []
        if late_days > 15:
            factors.append(f"High utility bill late days ({late_days} days)")
        if reload < 0.5:
            factors.append(f"Low mobile reload consistency ({reload:.2f})")
        if loans >= 2:
            factors.append(f"Multiple existing loans ({loans})")
        if factors:
            st.warning("**Contributing risk factors detected:**\n\n" + "\n".join(f"- {f}" for f in factors))

    # Risk meter
    st.progress(risk_probability)
    st.caption(f"Risk Score: {risk_probability * 100:.1f}%")

    with st.expander("📄 View submitted applicant data"):
        st.dataframe(input_features)

# --- Footer ---
st.markdown("---")
st.caption("Developed by Charuka Bandara | AI-Powered Financial Inclusion Project")
