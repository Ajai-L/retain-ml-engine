import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Model artifact location
MODEL_PATH = "models/retain_rf_model.pkl"

app = FastAPI(
    title="R.E.T.A.I.N. ML Engine Microservice",
    description="Proactive employee attrition & flight risk evaluation API for React frontend integration.",
    version="1.0.0"
)

# Enable CORS middleware to allow cross-origin requests from React local dev server (e.g., localhost:3000, localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to cache model artifacts in memory
model_artifacts: Dict[str, Any] = {}



def load_model_bundle() -> Dict[str, Any]:
    """Loads the serialized Random Forest model bundle and encoders into memory."""
    global model_artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Serialized model bundle not found at '{MODEL_PATH}'. "
            "Ensure 'src/model_training.py' has been executed."
        )
    model_artifacts = joblib.load(MODEL_PATH)
    print(f"[API Gateway] Model loaded successfully from '{MODEL_PATH}'.")
    return model_artifacts


@app.on_event("startup")
def startup_event():
    """FastAPI startup hook to load model on application launch."""
    try:
        load_model_bundle()
    except Exception as e:
        print(f"[API Gateway Warning] Model not loaded at startup: {e}")


class PredictionResponse(BaseModel):
    flight_risk_probability: float = Field(..., description="Calculated probability of attrition (0.0 to 1.0)")
    risk_level: str = Field(..., description="Categorized risk tier: LOW, MEDIUM, or HIGH")
    top_stressors: List[str] = Field(..., description="Top behavioral/demographic stress factors contributing to flight risk")


def derive_top_stressors(
    input_data: Dict[str, Any],
    feature_importances: Dict[str, float],
    top_n: int = 3
) -> List[str]:
    """
    Identifies top employee stressors based on feature importance and employee risk factors.
    """
    # Negative/risk indicator conditions for key features
    risk_triggers = {
        "OverTime": lambda v: str(v).lower() in ["yes", "1", "true"],
        "WorkLifeBalance": lambda v: float(v) <= 2,
        "JobSatisfaction": lambda v: float(v) <= 2,
        "EnvironmentSatisfaction": lambda v: float(v) <= 2,
        "JobInvolvement": lambda v: float(v) <= 2,
        "RelationshipSatisfaction": lambda v: float(v) <= 2,
        "DistanceFromHome": lambda v: float(v) >= 15,
        "YearsSinceLastPromotion": lambda v: float(v) >= 4,
        "NumCompaniesWorked": lambda v: float(v) >= 4,
        "MonthlyIncome": lambda v: float(v) < 4000,
        "PercentSalaryHike": lambda v: float(v) <= 12,
        "Income_per_JobLevel": lambda v: float(v) < 2000,
        "Job_Hopping_Index": lambda v: float(v) >= 0.50,
        "Satisfaction_Score": lambda v: float(v) <= 2.25,
        "Tenure_Ratio": lambda v: float(v) <= 0.25,
    }

    triggered_stressors = []
    for feature, imp in feature_importances.items():
        if feature in input_data:
            val = input_data[feature]
            if feature in risk_triggers:
                try:
                    if risk_triggers[feature](val):
                        triggered_stressors.append((feature, imp))
                except (ValueError, TypeError):
                    pass
            else:
                # Default heuristic for remaining features
                triggered_stressors.append((feature, imp))

    # Sort triggered stressors by feature importance descending
    triggered_stressors.sort(key=lambda x: x[1], reverse=True)
    
    top_stressors = [feat for feat, _ in triggered_stressors[:top_n]]
    
    # Fallback to top important features if no specific risk trigger was met
    if not top_stressors:
        top_stressors = list(feature_importances.keys())[:top_n]
        
    return top_stressors


def validate_and_sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and sanitizes employee payload for data coherence and logical consistency:
    - Enforces logical tenure bounds (YearsInCurrentRole <= YearsAtCompany <= TotalWorkingYears).
    """
    sanitized = dict(payload)

    try:
        total_years = float(sanitized.get("TotalWorkingYears", 0.0))
        years_at_co = float(sanitized.get("YearsAtCompany", 0.0))
        years_in_role = float(sanitized.get("YearsInCurrentRole", 0.0))
        years_curr_mgr = float(sanitized.get("YearsWithCurrManager", 0.0))

        if years_at_co > total_years and total_years > 0:
            years_at_co = total_years
            sanitized["YearsAtCompany"] = years_at_co

        if years_in_role > years_at_co and years_at_co > 0:
            sanitized["YearsInCurrentRole"] = years_at_co

        if years_curr_mgr > years_at_co and years_at_co > 0:
            sanitized["YearsWithCurrManager"] = years_at_co
    except (ValueError, TypeError):
        pass

    return sanitized


def evaluate_boundary_conditions(
    payload: Dict[str, Any], raw_proba: float
) -> Tuple[float, List[str]]:
    """
    Evaluates hard domain boundary conditions (Points-of-No-Return) and coupled cross-feature multipliers.
    Certain extreme conditions (e.g. 100km commute, starvation wages, chronic overtime burnout)
    act as non-negotiable flight triggers regardless of other positive factors.
    """
    try:
        distance = float(payload.get("DistanceFromHome", 0.0))
        monthly_income = float(payload.get("MonthlyIncome", 5000.0))
        job_level = float(payload.get("JobLevel", 1.0))
        wlb = float(payload.get("WorkLifeBalance", 3.0))
        env_sat = float(payload.get("EnvironmentSatisfaction", 3.0))
        job_sat = float(payload.get("JobSatisfaction", 3.0))
        rel_sat = float(payload.get("RelationshipSatisfaction", 3.0))
        overtime = str(payload.get("OverTime", "")).lower() in ["yes", "1", "true"]
        num_companies = float(payload.get("NumCompaniesWorked", 0.0))
        total_years = float(payload.get("TotalWorkingYears", 0.0))
        years_company = float(payload.get("YearsAtCompany", 0.0))
        years_no_promo = float(payload.get("YearsSinceLastPromotion", 0.0))
        stock_options = float(payload.get("StockOptionLevel", 0.0))
    except (ValueError, TypeError):
        return raw_proba, []

    job_hopping = num_companies / (total_years + 1.0)
    income_per_level = monthly_income / (job_level + 1.0)
    satisfaction_avg = (job_sat + env_sat + rel_sat + wlb) / 4.0

    dealbreakers: List[str] = []
    compound_penalties = 0.0
    floor_probability = 0.0

    # =========================================================
    # 1. HARD BOUNDARY CONDITIONS (Points-of-No-Return)
    # =========================================================
    
    # A. Commute Infeasibility Boundary
    if distance >= 80:
        floor_probability = max(floor_probability, 0.92)
        dealbreakers.append("Critical Commute Infeasibility (80km+ Unsustainable Daily Travel)")
    elif distance >= 50:
        floor_probability = max(floor_probability, 0.65)
        dealbreakers.append("Severe Commute Distance (50km+ Daily Travel)")

    # B. Extreme Wage Deprivation & Exploitation Boundary
    if monthly_income < 1000:
        floor_probability = max(floor_probability, 0.98)
        dealbreakers.append("Severe Wage Deprivation (Substandard/Unlivable Monthly Income)")
    elif income_per_level < 1200 and job_level >= 2:
        floor_probability = max(floor_probability, 0.85)
        dealbreakers.append("Extreme Role Underpayment (Senior Role on Sub-Standard Pay)")

    # C. Chronic Overtime Burnout & Work-Life Breakdown
    if overtime and wlb <= 1:
        floor_probability = max(floor_probability, 0.90)
        dealbreakers.append("Chronic Overtime Burnout & Severe Work-Life Imbalance")

    # D. Toxic Workplace Culture & Multi-Pillar Dissatisfaction
    if env_sat <= 1 and job_sat <= 1 and wlb <= 2:
        floor_probability = max(floor_probability, 0.88)
        dealbreakers.append("Toxic Workplace Environment & Severe Burnout")

    # =========================================================
    # 2. COUPLED CROSS-FEATURE INTERACTIONS (Multipliers)
    # =========================================================

    # Overtime + Long Commute fatigue compounding
    if overtime and distance >= 35:
        compound_penalties += 0.25
        dealbreakers.append("Compounded Overtime Fatigue with Long Daily Commute")

    # Uncompensated low-wage overtime
    if overtime and monthly_income < 3500 and monthly_income >= 1000:
        compound_penalties += 0.20
        dealbreakers.append("Uncompensated Overtime with Below-Average Wage")

    # High mobility worker facing dissatisfaction
    if job_hopping >= 0.60 and (job_sat <= 2 or env_sat <= 2):
        compound_penalties += 0.20
        dealbreakers.append("High Mobility Turnover Triggered by Dissatisfaction")

    # Career stagnation / Dead-end role
    if years_no_promo >= 7 and job_level <= 2 and years_company >= 6:
        compound_penalties += 0.25
        dealbreakers.append("Career Dead-End & Extended Promotion Stagnation")

    # Retention Handcuffs Buffer (Only applies if no extreme hard dealbreaker is active)
    if floor_probability < 0.80:
        if monthly_income >= 12000 and stock_options >= 2 and satisfaction_avg >= 3.0:
            compound_penalties -= 0.15

    # Compute final probability combining raw model score with boundary floors and compound penalties
    combined_proba = min(0.99, max(raw_proba + compound_penalties, floor_probability))
    return combined_proba, dealbreakers


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for microservice monitoring."""
    model_loaded = "model" in model_artifacts
    return {
        "status": "UP" if model_loaded else "DEGRADED",
        "service": "RETAIN ML Engine API",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_flight_risk(payload: Dict[str, Any]):
    """
    Accepts employee behavioral and demographic metrics in JSON format,
    validates data coherence, evaluates non-negotiable boundary conditions and coupled interactions,
    and returns calibrated flight risk probabilities and primary workplace stressors.
    """
    if not model_artifacts:
        try:
            load_model_bundle()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model bundle unavailable: {str(e)}"
            )

    sanitized_payload = validate_and_sanitize_payload(payload)

    model = model_artifacts["model"]
    feature_names = model_artifacts["feature_names"]
    encoders = model_artifacts.get("encoders", {})
    feature_importances = model_artifacts.get("feature_importances", {})

    # Build feature row matching model training schema
    row = {}
    for feat in feature_names:
        if feat in sanitized_payload:
            raw_val = sanitized_payload[feat]
            # Encode categorical strings if an encoder exists
            if feat in encoders:
                le = encoders[feat]
                raw_str = str(raw_val)
                if raw_str in le.classes_:
                    row[feat] = int(le.transform([raw_str])[0])
                else:
                    # Fallback for unseen categorical values
                    row[feat] = 0
            else:
                try:
                    row[feat] = float(raw_val)
                except (ValueError, TypeError):
                    row[feat] = 0.0

    # Dynamic Feature Engineering Ratios
    monthly_income = float(row.get("MonthlyIncome", 0.0))
    job_level = float(row.get("JobLevel", 1.0))
    num_companies = float(row.get("NumCompaniesWorked", 0.0))
    total_years = float(row.get("TotalWorkingYears", 0.0))
    years_at_co = float(row.get("YearsAtCompany", 0.0))
    job_sat = float(row.get("JobSatisfaction", 3.0))
    env_sat = float(row.get("EnvironmentSatisfaction", 3.0))
    rel_sat = float(row.get("RelationshipSatisfaction", 3.0))
    wlb = float(row.get("WorkLifeBalance", 3.0))

    row["Income_per_JobLevel"] = monthly_income / (job_level + 1.0)
    row["Job_Hopping_Index"] = num_companies / (total_years + 1.0)
    row["Satisfaction_Score"] = (job_sat + env_sat + rel_sat + wlb) / 4.0
    row["Tenure_Ratio"] = years_at_co / (total_years + 1.0)

    # Pass enriched input to derive_top_stressors
    enriched_payload = dict(sanitized_payload)
    enriched_payload["Income_per_JobLevel"] = row["Income_per_JobLevel"]
    enriched_payload["Job_Hopping_Index"] = row["Job_Hopping_Index"]
    enriched_payload["Satisfaction_Score"] = row["Satisfaction_Score"]
    enriched_payload["Tenure_Ratio"] = row["Tenure_Ratio"]

    df_input = pd.DataFrame([row], columns=feature_names)

    # Predict raw model attrition probability
    raw_proba = float(model.predict_proba(df_input)[0][1])

    # Evaluate boundary conditions and coupled interaction penalties
    final_proba, dealbreaker_stressors = evaluate_boundary_conditions(sanitized_payload, raw_proba)
    proba_rounded = round(final_proba, 4)

    # Determine Risk Tier (Calibrated against dataset baseline attrition rate of 16.1%)
    if proba_rounded >= 0.35:
        risk_tier = "HIGH"
    elif proba_rounded >= 0.16:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # Derive Top Stressors from feature importances and active dealbreaker conditions
    stressors = derive_top_stressors(enriched_payload, feature_importances, top_n=3)

    # Merge dealbreaker stressors at the front of the list for clear visibility
    merged_stressors: List[str] = []
    for d in dealbreaker_stressors:
        if d not in merged_stressors:
            merged_stressors.append(d)
            
    for s in stressors:
        if s not in merged_stressors and len(merged_stressors) < 3:
            merged_stressors.append(s)

    final_top_stressors = merged_stressors[:3] if merged_stressors else ["Normal Workplace Variance"]

    return PredictionResponse(
        flight_risk_probability=proba_rounded,
        risk_level=risk_tier,
        top_stressors=final_top_stressors
    )




if __name__ == "__main__":
    import uvicorn
    print("[API Gateway] Starting Uvicorn server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
