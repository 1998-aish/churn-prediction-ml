"""Streamlit demo: enter a customer's details and get a churn-risk score.

Run locally:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import load_model, predict_one  # noqa: E402

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.caption("Logistic regression trained on 7,043 telecom customers · "
           "[Source code](https://github.com/1998-aish/churn-prediction-ml)")

threshold = load_model()["threshold"]

st.subheader("Account")
c1, c2 = st.columns(2)
tenure = c1.slider("Tenure (months)", 0, 72, 6)
contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly = c1.number_input("Monthly charges ($)", 18.0, 120.0, 75.0, step=1.0)
payment = c2.selectbox("Payment method", ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])
paperless = c1.radio("Paperless billing", ["Yes", "No"], horizontal=True)

st.subheader("Services")
c3, c4 = st.columns(2)
internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
phone = c4.radio("Phone service", ["Yes", "No"], horizontal=True)

no_net = internet == "No"
addon_options = ["No internet service"] if no_net else ["No", "Yes"]
security = c3.selectbox("Online security", addon_options)
backup = c4.selectbox("Online backup", addon_options)
protection = c3.selectbox("Device protection", addon_options)
support = c4.selectbox("Tech support", addon_options)
tv = c3.selectbox("Streaming TV", addon_options)
movies = c4.selectbox("Streaming movies", addon_options)
lines = c3.selectbox("Multiple lines", ["No phone service"] if phone == "No" else ["No", "Yes"])

st.subheader("Customer")
c5, c6 = st.columns(2)
gender = c5.radio("Gender", ["Female", "Male"], horizontal=True)
senior = c6.radio("Senior citizen", ["No", "Yes"], horizontal=True)
partner = c5.radio("Partner", ["No", "Yes"], horizontal=True)
dependents = c6.radio("Dependents", ["No", "Yes"], horizontal=True)

customer = {
    "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": monthly * max(tenure, 1),
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents, "PhoneService": phone,
    "MultipleLines": lines, "InternetService": internet, "OnlineSecurity": security,
    "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
    "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
}

result = predict_one(customer)
prob = result["churn_probability"]

st.divider()
st.metric("Churn probability", f"{prob:.0%}")
st.progress(min(float(prob), 1.0))
if result["risk"] == "High":
    st.error("**High risk:** prioritize for a retention offer.")
elif result["risk"] == "Medium":
    st.warning("**Medium risk:** include in the retention campaign.")
else:
    st.success("**Low risk:** no action needed.")

tips = []
if contract == "Month-to-month":
    tips.append("Offer a discounted 1- or 2-year contract (month-to-month customers churn at 43% vs 3% on two-year).")
if tenure <= 12:
    tips.append("New customer: add an onboarding check-in (47% of first-year customers churn).")
if payment == "Electronic check":
    tips.append("Encourage automatic payment (electronic-check customers churn at 45% vs ~16% on auto-pay).")
if internet != "No" and support == "No":
    tips.append("Offer a free tech-support trial.")
if tips and result["risk"] != "Low":
    st.markdown("**Suggested retention actions**")
    for t in tips:
        st.markdown(f"- {t}")

st.caption(f"Decision threshold: {threshold:.2f}, tuned to catch about 80% of churners.")
