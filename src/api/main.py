# src/api/main.py
import joblib
import shap
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- Chargement au démarrage (une seule fois) ---
model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = joblib.load("models/churn_pipeline.pkl")
    preprocessor = pipeline.named_steps["preprocessor"]
    model        = pipeline.named_steps["model"]
    explainer    = shap.LinearExplainer(model, masker=shap.maskers.Independent(
                       np.zeros((1, len(pipeline.feature_names_in_))), max_samples=100))
    model_store["pipeline"]     = pipeline
    model_store["preprocessor"] = preprocessor
    model_store["explainer"]    = explainer
    yield
    model_store.clear()

app = FastAPI(title="Churn Prediction API", lifespan=lifespan)

# --- Schéma d'entrée ---
class CustomerFeatures(BaseModel):
    SeniorCitizen: int           = Field(..., ge=0, le=1)
    Partner: int                 = Field(..., ge=0, le=1)
    Dependents: int              = Field(..., ge=0, le=1)
    tenure: int                  = Field(..., ge=0)
    PhoneService: int            = Field(..., ge=0, le=1)
    PaperlessBilling: int        = Field(..., ge=0, le=1)
    MonthlyCharges: float        = Field(..., gt=0)
    TotalCharges: float          = Field(..., ge=0)
    NbServices: int              = Field(..., ge=0, le=6)
    AvgMonthlyCharge: float      = Field(..., ge=0)
    gender_Male: int             = Field(..., ge=0, le=1)
    MultipleLines_No_phone_service: int = Field(..., ge=0, le=1, alias="MultipleLines_No phone service")
    MultipleLines_Yes: int       = Field(..., ge=0, le=1)
    InternetService_Fiber_optic: int = Field(..., ge=0, le=1, alias="InternetService_Fiber optic")
    InternetService_No: int      = Field(..., ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(..., ge=0, le=1, alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int      = Field(..., ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(..., ge=0, le=1, alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int        = Field(..., ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(..., ge=0, le=1, alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int    = Field(..., ge=0, le=1)
    TechSupport_No_internet_service: int = Field(..., ge=0, le=1, alias="TechSupport_No internet service")
    TechSupport_Yes: int         = Field(..., ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(..., ge=0, le=1, alias="StreamingTV_No internet service")
    StreamingTV_Yes: int         = Field(..., ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(..., ge=0, le=1, alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: int     = Field(..., ge=0, le=1)
    Contract_One_year: int       = Field(..., ge=0, le=1, alias="Contract_One year")
    Contract_Two_year: int       = Field(..., ge=0, le=1, alias="Contract_Two year")
    PaymentMethod_Credit_card: int = Field(..., ge=0, le=1, alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Mailed check")
    TenureGroup_0_12m: int       = Field(..., ge=0, le=1, alias="TenureGroup_0-12m")
    TenureGroup_13_24m: int      = Field(..., ge=0, le=1, alias="TenureGroup_13-24m")
    TenureGroup_25_48m: int      = Field(..., ge=0, le=1, alias="TenureGroup_25-48m")
    TenureGroup_49m: int         = Field(..., ge=0, le=1, alias="TenureGroup_49m+")

    model_config = {"populate_by_name": True}

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerFeatures):
    df = pd.DataFrame([customer.model_dump(by_alias=True)])

    proba = model_store["pipeline"].predict_proba(df)[0, 1]

    # SHAP : top 3 features
    preprocessor = model_store["preprocessor"]
    explainer    = model_store["explainer"]
    X_transformed = preprocessor.transform(df)
    shap_vals     = explainer(X_transformed)
    feature_names = (
        ["tenure", "MonthlyCharges", "TotalCharges", "NbServices"] +
        [c for c in df.columns if c not in ["tenure", "MonthlyCharges", "TotalCharges", "NbServices"]]
    )
    shap_dict = dict(zip(feature_names, shap_vals.values[0]))
    top3 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction":  int(proba >= 0.5),
        "top_factors":       [{"feature": k, "shap_value": round(float(v), 4)} for k, v in top3]
    }
