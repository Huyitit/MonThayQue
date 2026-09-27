"""
FastAPI REST Service for House Price Prediction (Application 2).
Provides real-time property valuation inference, health checks, model metadata,
and serves the mobile-responsive web user interface.
"""
import sys
import json
import argparse
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
setattr(sys.modules.get("__main__"), "HouseFeatureEngineer", features.HouseFeatureEngineer)

try:
    from schemas import (
        HouseInputSchema,
        ValuationResponseSchema,
        ValuationRangeSchema,
        HealthResponseSchema,
        ModelInfoSchema,
    )
except ImportError:
    from .schemas import (
        HouseInputSchema,
        ValuationResponseSchema,
        ValuationRangeSchema,
        HealthResponseSchema,
        ModelInfoSchema,
    )

# Initialize FastAPI App
app = FastAPI(
    title="Vietnam House Price Valuation REST API",
    description=(
        "Intelligent System Service for Residential Real Estate Valuation in Vietnam. "
        "Transforms property features via Scikit-Learn pipeline and serves real-time regression inference."
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
RMSE_DEFAULT = 1.5838


def load_artifacts():
    """Load model pipeline and metadata safely into memory."""
    global PIPELINE, METADATA, RMSE_DEFAULT
    pipeline_path = MODEL_DIR / "pipeline.joblib"
    metadata_path = MODEL_DIR / "metadata.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}")
    
    PIPELINE = joblib.load(pipeline_path)
    
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
            RMSE_DEFAULT = METADATA.get("metrics", {}).get("test_rmse_billion_vnd", 1.5838)
    else:
        METADATA = {
            "champion_model": "Gradient Boosting Regressor",
            "target": {"name": "Price", "unit": "Billion VND"},
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
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoSchema, tags=["Model"])
def model_info():
    """Get metadata about the champion model, features, and evaluation metrics."""
    if not METADATA:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable")
    return ModelInfoSchema(
        application=METADATA.get("application", "House Price Prediction"),
        champion_model=METADATA.get("champion_model", "Gradient Boosting Regressor"),
        target=METADATA.get("target", {"name": "Price", "unit": "Billion VND"}),
        input_features=METADATA.get("input_features", {
            "numerical": features.NUMERICAL_FEATURES,
            "categorical": features.CATEGORICAL_FEATURES
        }),
        engineered_features=METADATA.get("engineered_features", features.ENGINEERED_NUMERICAL_FEATURES),
        transformed_feature_dimension=METADATA.get("transformed_feature_dimension", 23),
        top_provinces=METADATA.get("top_provinces", features.TOP_PROVINCES),
        metrics=METADATA.get("metrics", {})
    )


@app.post("/predict", response_model=ValuationResponseSchema, tags=["Inference"])
def predict(payload: HouseInputSchema):
    """
    Execute residential valuation pipeline on property features.
    
    Workflow:
    Raw JSON Input -> Pydantic Validation -> DataFrame -> Pipeline Preprocessing -> Gradient Boosting -> Structured Valuation Response
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
        raw_pred = float(PIPELINE.predict(input_df)[0])
        # Prevent non-physical negative valuation (if any edge outlier occurs)
        bounded_pred = max(0.2, raw_pred)

        # Compute formatted currency string (Billion VND -> VND)
        vnd_price = int(bounded_pred * 1_000_000_000)
        formatted_vnd = f"{vnd_price:,} ₫"

        # Derived metrics
        price_per_m2_million = round(float((bounded_pred * 1_000.0) / payload.area), 2)

        # Valuation confidence band based on test RMSE
        rmse = float(METADATA.get("metrics", {}).get("test_rmse_billion_vnd", RMSE_DEFAULT))
        low_estimate = round(max(0.1, bounded_pred - rmse), 3)
        high_estimate = round(bounded_pred + rmse, 3)

        # Domain engineered features for model explainability & transparency
        total_floor_area = round(float(payload.area * payload.floors), 2)
        bed_bath_ratio = round(float(payload.bedrooms / (payload.bathrooms + 1.0)), 3)
        room_density = round(float((payload.bedrooms + payload.bathrooms) / (payload.area + 1e-5)), 4)

        return ValuationResponseSchema(
            predicted_price_billion=round(bounded_pred, 3),
            formatted_price_vnd=formatted_vnd,
            price_per_m2_million=price_per_m2_million,
            valuation_range=ValuationRangeSchema(
                low_estimate_billion=low_estimate,
                high_estimate_billion=high_estimate,
                confidence_interval_note=f"Empirical 68% confidence band derived from test RMSE (±{rmse:.2f} Billion VND)."
            ),
            engineered_features={
                "Total_Floor_Area": total_floor_area,
                "Bed_Bath_Ratio": bed_bath_ratio,
                "Room_Density": room_density,
            },
            champion_model=METADATA.get("champion_model", "Gradient Boosting Regressor"),
            model_metrics=METADATA.get("metrics", {})
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
    parser = argparse.ArgumentParser(description="House Price Valuation REST API & Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    args = parser.parse_args()

    print(f"Starting House Price Valuation API & Web App on http://{args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port)
