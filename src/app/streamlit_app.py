# src/app/streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("🔮 Churn Prediction")
st.markdown("Renseignez les informations du client pour estimer son risque de churn.")

# --- Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    tenure          = st.number_input("Ancienneté (mois)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Charges mensuelles (€)", min_value=0.0, value=50.0)
    total_charges   = st.number_input("Charges totales (€)", min_value=0.0, value=600.0)
    nb_services     = st.slider("Nombre de services", 0, 6, 2)

with col2:
    senior          = st.selectbox("Senior Citizen", [0, 1])
    partner         = st.selectbox("Partner", [0, 1])
    dependents      = st.selectbox("Dependents", [0, 1])
    phone_service   = st.selectbox("Phone Service", [0, 1])
    paperless       = st.selectbox("Paperless Billing", [0, 1])

with col3:
    contract        = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment         = st.selectbox("Payment Method", [
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                        "Electronic check",
                        "Mailed check"
                      ])
    gender_male     = st.selectbox("Genre (Male=1)", [0, 1])

# --- Construction du payload ---
avg_monthly = total_charges / (tenure + 1)

payload = {
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "PaperlessBilling": paperless,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "NbServices": nb_services,
    "AvgMonthlyCharge": avg_monthly,
    "gender_Male": gender_male,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,
    "OnlineSecurity_No internet service": 1 if internet == "No" else 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 1 if internet == "No" else 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 1 if internet == "No" else 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 1 if internet == "No" else 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 1 if internet == "No" else 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_No internet service": 1 if internet == "No" else 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
    "TenureGroup_0-12m": 1 if tenure <= 12 else 0,
    "TenureGroup_13-24m": 1 if 13 <= tenure <= 24 else 0,
    "TenureGroup_25-48m": 1 if 25 <= tenure <= 48 else 0,
    "TenureGroup_49m+": 1 if tenure >= 49 else 0,
}

# --- Prédiction ---
if st.button("Prédire"):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result   = response.json()

        proba = result["churn_probability"]
        pred  = result["churn_prediction"]

        st.divider()
        color = "🔴" if pred == 1 else "🟢"
        st.metric(f"{color} Probabilité de churn", f"{proba*100:.1f}%")

        st.subheader("Facteurs principaux")
        for factor in result["top_factors"]:
            direction = "↑ risque" if factor["shap_value"] > 0 else "↓ risque"
            st.write(f"**{factor['feature']}** : {factor['shap_value']:+.4f} ({direction})")

    except Exception as e:
        st.error(f"Erreur API : {e}")