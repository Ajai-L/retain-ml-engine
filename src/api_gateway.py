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


def evaluate_independent_stressor_penalties(payload: Dict[str, Any]) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Evaluates features independently to identify extreme stressor outliers and human domain anomalies.
    Returns a risk penalty score boost and a list of triggered critical stressor flags.
    """
    try:
        distance = float(payload.get("DistanceFromHome", 0.0))
        monthly_income = float(payload.get("MonthlyIncome", 5000.0))
        job_level = float(payload.get("JobLevel", 1.0))
        wlb = float(payload.get("WorkLifeBalance", 3.0))
        env_sat = float(payload.get("EnvironmentSatisfaction", 3.0))
        job_sat = float(payload.get("JobSatisfaction", 3.0))
        overtime = str(payload.get("OverTime", "")).lower() in ["yes", "1", "true"]
        num_companies = float(payload.get("NumCompaniesWorked", 0.0))
        total_years = float(payload.get("TotalWorkingYears", 0.0))
        years_no_promo = float(payload.get("YearsSinceLastPromotion", 0.0))
    except (ValueError, TypeError):
        return 0.0, []

    job_hopping = num_companies / (total_years + 1.0)
    income_per_level = monthly_income / (job_level + 1.0)

    penalty = 0.0
    critical_flags: List[Tuple[str, float]] = []

    # 1. Distance From Home (Extreme Commute Stressor, e.g. 100km)
    if distance >= 80:
        penalty += 0.55
        critical_flags.append(("DistanceFromHome (Extreme Commute Distance)", 0.99))
    elif distance >= 50:
        penalty += 0.35
        critical_flags.append(("DistanceFromHome (Severe Commute Distance)", 0.85))
    elif distance >= 30:
        penalty += 0.20
        critical_flags.append(("DistanceFromHome (Long Commute)", 0.70))

    # 2. Extreme Salary Disparity
    if income_per_level < 1000:
        penalty += 0.40
        critical_flags.append(("MonthlyIncome (Severe Role Underpayment)", 0.95))
    elif income_per_level < 1800:
        penalty += 0.20
        critical_flags.append(("MonthlyIncome (Below Average Compensation)", 0.75))

    # 3. Work-Life Balance
    if wlb <= 1:
        penalty += 0.25
        critical_flags.append(("WorkLifeBalance (Severe Work-Life Imbalance)", 0.85))

    # 4. Environment & Job Dissatisfaction
    if env_sat <= 1:
        penalty += 0.20
        critical_flags.append(("EnvironmentSatisfaction (Poor Work Environment)", 0.80))
    if job_sat <= 1:
        penalty += 0.20
        critical_flags.append(("JobSatisfaction (Low Job Satisfaction)", 0.80))

    # 5. OverTime
    if overtime:
        penalty += 0.15
        critical_flags.append(("OverTime (Mandatory Overtime Stress)", 0.75))

    # 6. Job Hopping Velocity
    if job_hopping >= 0.8:
        penalty += 0.25
        critical_flags.append(("Job_Hopping_Index (High Historical Turnover)", 0.85))

    # 7. Promotion Stagnation
    if years_no_promo >= 6:
        penalty += 0.20
        critical_flags.append(("YearsSinceLastPromotion (Career Stagnation)", 0.75))

    return penalty, critical_flags


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
    validates data coherence, independently evaluates extreme stressor anomalies,
    and returns calibrated flight risk probabilities and top behavioral stressors.
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

    # Evaluate independent extreme stressor penalties
    stressor_penalty, critical_flags = evaluate_independent_stressor_penalties(sanitized_payload)

    # Final Probability blending raw model score with independent stressor penalties
    proba = min(0.99, max(raw_proba, raw_proba + stressor_penalty))
    proba_rounded = round(proba, 4)

    # Determine Risk Tier (Calibrated against dataset baseline attrition rate of 16.1%)
    if proba_rounded >= 0.35:
        risk_tier = "HIGH"
    elif proba_rounded >= 0.16:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # Derive Top Stressors
    stressors = derive_top_stressors(enriched_payload, feature_importances, top_n=3)

    # Prepend critical stressor flags if triggered by extreme independent values
    for flag_desc, _ in critical_flags:
        clean_name = flag_desc.split()[0]
        if clean_name not in stressors:
            stressors.insert(0, clean_name)
    stressors = stressors[:3]

    return PredictionResponse(
        flight_risk_probability=proba_rounded,
        risk_level=risk_tier,
        top_stressors=stressors
    )




if __name__ == "__main__":
    import uvicorn
    print("[API Gateway] Starting Uvicorn server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
