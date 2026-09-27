"""
Pydantic data validation schemas for Diabetes Prediction API.
Adheres to Assignment 02 Appendix C & D specifications.
Supports both standard snake_case and dataset PascalCase aliases.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class DiabetesInputSchema(BaseModel):
    model_config = ConfigDict()

    pregnancies: int = Field(
        default=1,
        ge=0,
        alias="Pregnancies",
        description="Number of times pregnant"
    )
    glucose: float = Field(
        default=120.0,
        ge=0.0,
        alias="Glucose",
        description="Plasma glucose concentration at 2 hours in an oral glucose tolerance test (mg/dL)"
    )
    blood_pressure: float = Field(
        default=70.0,
        ge=0.0,
        alias="BloodPressure",
        description="Diastolic blood pressure (mm Hg)"
    )
    skin_thickness: float = Field(
        default=20.0,
        ge=0.0,
        alias="SkinThickness",
        description="Triceps skin fold thickness (mm)"
    )
    insulin: float = Field(
        default=79.0,
        ge=0.0,
        alias="Insulin",
        description="2-Hour serum insulin (mu U/ml)"
    )
    bmi: float = Field(
        default=25.0,
        ge=0.0,
        alias="BMI",
        description="Body mass index (weight in kg/(height in m)^2)"
    )
    diabetes_pedigree_function: float = Field(
        default=0.45,
        ge=0.0,
        alias="DiabetesPedigreeFunction",
        description="Diabetes pedigree function (genetic score)"
    )
    age: int = Field(
        default=30,
        ge=0,
        alias="Age",
        description="Age in years"
    )

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching original model feature names."""
        return {
            "Pregnancies": self.pregnancies,
            "Glucose": self.glucose,
            "BloodPressure": self.blood_pressure,
            "SkinThickness": self.skin_thickness,
            "Insulin": self.insulin,
            "BMI": self.bmi,
            "DiabetesPedigreeFunction": self.diabetes_pedigree_function,
            "Age": self.age,
        }


class PredictionResponseSchema(BaseModel):
    prediction: int = Field(description="Binary classification outcome (0: Non-Diabetic, 1: Diabetic)")
    label: str = Field(description="Human-readable outcome label ('Diabetic' or 'Non-Diabetic')")
    probability: float = Field(description="Posterior probability of diabetes (Class 1)")
    confidence: float = Field(description="Model confidence score for the winning class")
    risk_level: str = Field(description="Clinical triage risk category (Low Risk, Moderate Risk, High Risk)")
    interpretation: str = Field(description="Concise clinical interpretation and actionable advice")
    engineered_features: Dict[str, float] = Field(
        description="Domain-engineered features computed during inference pipeline execution"
    )
    champion_model: str = Field(description="Model architecture serving inference")


class HealthResponseSchema(BaseModel):
    status: str = Field(description="API service health status")
    model_loaded: bool = Field(description="Whether the ML inference pipeline is loaded in memory")
    version: str = Field(description="API version")


class ModelInfoSchema(BaseModel):
    application: str
    champion_model: str
    input_features: list[str]
    engineered_features: list[str]
    total_feature_count: int
    metrics: Dict[str, Any]
