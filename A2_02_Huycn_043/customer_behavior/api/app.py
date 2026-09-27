"""
FastAPI REST Service for E-Commerce Customer Behavior & Interest Discovery (Application 3).
Provides real-time multimodal interest classification, health checks, model metadata,
and serves the mobile-responsive web user interface.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
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
setattr(sys.modules.get("__main__"), "CustomerFeatureEngineer", features.CustomerFeatureEngineer)

try:
    from schemas import (
        CustomerInputSchema,
        PredictionResponseSchema,
        DepartmentProbabilitySchema,
        HealthResponseSchema,
        ModelInfoSchema,
    )
except ImportError:
    from .schemas import (
        CustomerInputSchema,
        PredictionResponseSchema,
        DepartmentProbabilitySchema,
        HealthResponseSchema,
        ModelInfoSchema,
    )

# Initialize FastAPI App
app = FastAPI(
    title="E-Commerce Customer Behavior & Interest Discovery REST API",
    description=(
        "Intelligent System Service for Customer Product Interest Discovery. "
        "Transforms multimodal customer engagement metrics and unstructured review text "
        "via Scikit-Learn TF-IDF + Tabular pipeline and serves real-time department interest classification."
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
        with open(metadata_path, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
    else:
        METADATA = {
            "champion_model": "Multimodal Logistic Regression with L2 Regularization",
            "application": "Application 3: E-Commerce Customer Behavior (Interest Discovery)",
        }


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health", response_model=HealthResponseSchema, tags=["System"])
def health_check():
    """Service health check endpoint."""
    return HealthResponseSchema(
        status="healthy",
        model_loaded=PIPELINE is not None,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoSchema, tags=["Model"])
def model_info():
    """Get metadata about the champion model, features, classes, and evaluation metrics."""
    if not METADATA:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable")

    target_info = METADATA.get("target", {})
    classes = target_info.get("classes", ["Bottoms", "Dresses", "Intimate", "Jackets", "Tops"])
    dimensions = METADATA.get(
        "transformed_feature_dimension",
        {"text_dimension": 2500, "tabular_dimension": 6, "total_combined_dimension": 2506}
    )

    return ModelInfoSchema(
        application=METADATA.get("application", "Application 3: Customer Behavior"),
        domain=METADATA.get("domain", "E-Commerce Customer Product Interest Discovery"),
        champion_model=METADATA.get("champion_model", "Multimodal Logistic Regression"),
        model_family=METADATA.get("model_family", "Linear Classification (Multinomial Softmax)"),
        target_classes=classes,
        feature_dimensions=dimensions,
        metrics=METADATA.get("metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponseSchema, tags=["Inference"])
def predict(payload: CustomerInputSchema):
    """
    Execute multimodal customer interest discovery inference pipeline.

    Workflow:
    Raw JSON Payload -> Pydantic Validation -> DataFrame -> CustomerFeatureEngineer ->
    Multimodal ColumnTransformer (TF-IDF + StandardScaler) -> Multinomial Softmax Classifier ->
    Probabilities & Ranked Department Recommendations.
    """
    if PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference pipeline not loaded into memory",
        )

    try:
        # Convert validated payload to DataFrame matching training schema
        feature_dict = payload.to_feature_dict()
        input_df = pd.DataFrame([feature_dict])

        # Execute Inference Pipeline
        raw_pred = str(PIPELINE.predict(input_df)[0])
        prob_array = PIPELINE.predict_proba(input_df)[0]
        classes = [str(c) for c in PIPELINE.classes_]

        # Construct probability dictionary
        prob_dict: Dict[str, float] = {}
        ranked_list: List[DepartmentProbabilitySchema] = []

        for cls_name, prob_val in zip(classes, prob_array):
            prob_float = float(prob_val)
            prob_dict[cls_name] = round(prob_float, 4)
            ranked_list.append(
                DepartmentProbabilitySchema(
                    department=cls_name,
                    probability=round(prob_float, 4),
                    percentage=round(prob_float * 100.0, 2),
                )
            )

        # Sort ranked list descending by probability
        ranked_list.sort(key=lambda x: x.probability, reverse=True)
        confidence = float(max(prob_array))

        # Compute engineered features for transparency
        review_text = payload.review_text.strip()
        title_text = (payload.title or "").strip()
        word_count = len(review_text.split()) if review_text else 0
        review_length = len(review_text)
        log_pos_feedback = round(float(np.log1p(max(payload.positive_feedback_count, 0))), 4)
        upper_chars = sum(1 for c in review_text if c.isupper())
        upper_ratio = round(upper_chars / max(review_length, 1), 4)
        has_title = 1 if title_text else 0

        engineered_dict = {
            "word_count": word_count,
            "review_length": review_length,
            "log_positive_feedback": log_pos_feedback,
            "uppercase_ratio": upper_ratio,
            "has_title": has_title,
        }

        # Determine Customer Behavioral Profile & Targeted Action
        if payload.rating >= 4 and payload.recommended_ind == 1:
            behavior_category = "High-Intent Brand Promoter"
            interpretation = (
                f"Customer expresses strong enthusiasm for '{raw_pred}'. Recommend deploying VIP early-access "
                f"promotions, personalized {raw_pred} catalog lookbooks, and loyalty reward bonuses."
            )
        elif payload.rating <= 2 or payload.recommended_ind == 0:
            behavior_category = "Critical / Churn Risk Customer"
            interpretation = (
                f"Customer indicated dissatisfaction regarding '{raw_pred}' purchase. Immediate retention priority: "
                f"dispatch automated post-purchase survey, sizing/fit consultation offer, and satisfaction guarantee."
            )
        else:
            behavior_category = "Moderate / Selective Evaluator"
            interpretation = (
                f"Customer shows steady engagement with '{raw_pred}'. Target with social-proof styling guides, "
                f"top-rated complementary essentials, and limited-time category incentives."
            )

        return PredictionResponseSchema(
            predicted_department=raw_pred,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            ranked_departments=ranked_list,
            behavior_category=behavior_category,
            interpretation=interpretation,
            engineered_features=engineered_dict,
            champion_model=METADATA.get(
                "champion_model", "Multimodal Logistic Regression with L2 Regularization"
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )


# Mount static web directory to serve user interface
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Customer Behavior API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind (default: 8002)")
    args = parser.parse_args()

    print(f"Starting RetailSense Customer Behavior API on http://{args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port)
