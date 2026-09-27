"""
FastAPI REST Service for Diabetes Prediction (Application 1).
Provides real-time clinical assessment inference, health checks, model metadata,
and serves the mobile-responsive web user interface.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Setup path to import local feature engineering module
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR.parent / "model"
WEB_DIR = CURRENT_DIR.parent / "web"

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(CURRENT_DIR))

try:
    from model import features
except ImportError:
    # pyrefly: ignore [missing-import]
    import features

# Register in sys.modules so joblib unpickler can resolve 'features'
sys.modules["features"] = features
# Register transformer in __main__ to guarantee unpickling robustness
setattr(sys.modules.get("__main__"), "ClinicalFeatureEngineer", features.ClinicalFeatureEngineer)

try:
    from schemas import (
        DiabetesInputSchema,
        PredictionResponseSchema,
        HealthResponseSchema,
        ModelInfoSchema,
    )
except ImportError:
    from .schemas import (
        DiabetesInputSchema,
        PredictionResponseSchema,
        HealthResponseSchema,
        ModelInfoSchema,
    )

# Initialize FastAPI App
app = FastAPI(
    title="Diabetes Prediction REST API",
    description=(
        "Intelligent System Service for Type 2 Diabetes Risk Prediction. "
        "Transforms clinical/demographic features via Scikit-Learn pipeline and serves real-time inference."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external web and mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and metadata state
PIPELINE = None
METADATA: Dict[str, Any] = {}


def load_artifacts():
    """Load model pipeline and metadata safely into memory."""
    global PIPELINE, METADATA
    pipeline_path = MODEL_DIR / "pipeline.joblib"
    metadata_path = MODEL_DIR / "metadata.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}")
    
    PIPELINE = joblib.load(pipeline_path)
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            METADATA = json.load(f)
    else:
        METADATA = {"champion_model": "Random Forest Classifier"}


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health", response_model=HealthResponseSchema, tags=["System"])
def health_check():
    """Service health check endpoint."""
    return HealthResponseSchema(
        status="healthy",
        model_loaded=PIPELINE is not None,
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoSchema, tags=["Model"])
def model_info():
    """Get metadata about the champion model, features, and evaluation metrics."""
    if not METADATA:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable")
    return ModelInfoSchema(
        application=METADATA.get("application", "Diabetes Prediction"),
        champion_model=METADATA.get("champion_model", "Random Forest Classifier"),
        input_features=METADATA.get("input_features", features.FEATURE_NAMES),
        engineered_features=METADATA.get("engineered_features", ["Glucose_BMI_Interaction", "Insulin_Glucose_Ratio"]),
        total_feature_count=METADATA.get("total_feature_count", 10),
        metrics=METADATA.get("metrics", {})
    )


@app.post("/predict", response_model=PredictionResponseSchema, tags=["Inference"])
def predict(payload: DiabetesInputSchema):
    """
    Execute clinical prediction pipeline on patient features.
    
    Workflow:
    Raw JSON Input -> Pydantic Validation -> DataFrame -> Pipeline Preprocessing -> Random Forest -> Structured Response
    """
    if PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference pipeline not loaded"
        )

    try:
        # Convert Pydantic model to DataFrame matching training schema
        feature_dict = payload.to_feature_dict()
        input_df = pd.DataFrame([feature_dict])

        # Execute Inference Pipeline
        raw_pred = PIPELINE.predict(input_df)[0]
        prob_array = PIPELINE.predict_proba(input_df)[0]
        prob_diabetic = float(prob_array[1])
        confidence = float(max(prob_array))

        # Compute engineered features for transparency
        glucose_bmi = round(float(payload.glucose * payload.bmi), 2)
        insulin_glucose = round(float(payload.insulin / (payload.glucose + 1e-5)), 4)

        # Determine clinical risk categorization
        if prob_diabetic >= 0.65:
            risk_level = "High Risk"
            interpretation = (
                "High probability of diabetes indicated. Elevated glucose/metabolic indicators suggest immediate "
                "clinical diagnostic confirmation (HbA1c / Oral Glucose Tolerance Test)."
            )
        elif prob_diabetic >= 0.35:
            risk_level = "Moderate Risk"
            interpretation = (
                "Borderline metabolic indicators. Lifestyle intervention, glycemic monitoring, and scheduled "
                "follow-up assessment are recommended."
            )
        else:
            risk_level = "Low Risk"
            interpretation = (
                "Patient indicators are currently within typical non-diabetic ranges. Continue standard preventative "
                "health screening."
            )

        label = "Diabetic" if raw_pred == 1 else "Non-Diabetic"

        return PredictionResponseSchema(
            prediction=int(raw_pred),
            label=label,
            probability=round(prob_diabetic, 4),
            confidence=round(confidence, 4),
            risk_level=risk_level,
            interpretation=interpretation,
            engineered_features={
                "Glucose_BMI_Interaction": glucose_bmi,
                "Insulin_Glucose_Ratio": insulin_glucose,
            },
            champion_model=METADATA.get("champion_model", "Random Forest Classifier")
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )


# Mount static web directory to serve user interface
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    print("Starting Diabetes Prediction API & Web App on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

