import os
import joblib
import pandas as pd
import numpy as np
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


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Java Spring Boot microservice monitoring."""
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
    preprocesses inputs using stored label encoders, and returns flight risk probability
    along with top behavioral stressors.
    """
    if not model_artifacts:
        try:
            load_model_bundle()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model bundle unavailable: {str(e)}"
            )

    model = model_artifacts["model"]
    feature_names = model_artifacts["feature_names"]
    encoders = model_artifacts.get("encoders", {})
    feature_importances = model_artifacts.get("feature_importances", {})

    # Build feature row matching model training schema
    row = {}
    for feat in feature_names:
        if feat in payload:
            raw_val = payload[feat]
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
        else:
            # Default missing features to 0
            row[feat] = 0.0

    df_input = pd.DataFrame([row], columns=feature_names)

    # Predict attrition probability
    proba = float(model.predict_proba(df_input)[0][1])
    proba_rounded = round(proba, 4)

    # Determine Risk Tier
    if proba_rounded >= 0.60:
        risk_tier = "HIGH"
    elif proba_rounded >= 0.30:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # Derive Top Stressors
    stressors = derive_top_stressors(payload, feature_importances, top_n=3)

    return PredictionResponse(
        flight_risk_probability=proba_rounded,
        risk_level=risk_tier,
        top_stressors=stressors
    )


if __name__ == "__main__":
    import uvicorn
    print("[API Gateway] Starting Uvicorn server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
