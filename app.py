import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Streamlit Config (must be first!)
# -----------------------------
st.set_page_config(page_title="Mutual Fund Prediction", layout="centered")

# -----------------------------
# Load pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(r"C:\Users\Mayur\Documents\MO\mf_pipeline.pkl")

pipeline = load_pipeline()

# Capture expected feature names
#expected_features = pipeline.feature_names_in_
# expected_features = pipeline.feature_names_in_

# Temporary fix: manually define your feature list
expected_features = [
    "Age", 
    "Income", 
    "Holding Amount", 
    "InvestmentExperience", 
    "Gender", 
    "Profession", 
    "RiskTolerance", 
    "InvestmentGoal"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧮 Mutual Fund Buy Prediction App")
st.write("Enter customer details below to predict if they will buy a mutual fund.")

# ---- Collect user input ----
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (₹)", min_value=0, max_value=10_000_000, value=500000, step=10000)
holding = st.number_input("Current Holding Amount (₹)", min_value=0, max_value=50_000_000, value=100000, step=5000)
investment_exp = st.slider("Investment Experience (Years)", 0, 40, 5)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
profession = st.text_input("Profession", "Manager")
risk = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
goal = st.selectbox("Investment Goal", ["Growth", "Balanced", "Income", "Conservative"])

# ---- Predict button ----
if st.button("🔮 Predict"):
    # Raw input (match your training column names exactly!)
    user_df = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Holding Amount": holding,       # IMPORTANT: space matches training
        "InvestmentExperience": investment_exp,
        "Gender": gender,
        "Profession": profession,
        "RiskTolerance": risk,
        "InvestmentGoal": goal
    }])

    # -----------------------------
    # Align columns with pipeline
    # -----------------------------
    for col in expected_features:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[expected_features]

    # -----------------------------
    # Prediction
    # -----------------------------
    prob = pipeline.predict_proba(user_df)[0, 1]
    pred = pipeline.predict(user_df)[0]

    # ---- Results ----
    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {'🟢 Will Buy' if pred == 1 else '🔴 Will Not Buy'}")
    st.write(f"**Probability of Buying:** {prob:.2%}")

    # Debug info (optional)
    st.write("ℹ️ Debug: Expected features count:", len(expected_features))
