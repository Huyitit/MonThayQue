"""
Pydantic data validation schemas for House Price Prediction API (Application 2).
Adheres to Assignment 02 Appendix C & D specifications.
Supports both standard snake_case and dataset original feature names.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class HouseInputSchema(BaseModel):
    model_config = ConfigDict()

    area: float = Field(
        default=75.0,
        ge=10.0,
        le=2000.0,
        alias="Area",
        description="Total lot/land area in square meters (m²)"
    )
    frontage: float = Field(
        default=4.5,
        ge=1.0,
        le=100.0,
        alias="Frontage",
        description="Facade/frontage width in meters (m)"
    )
    access_road: float = Field(
        default=6.0,
        ge=0.5,
        le=100.0,
        alias="Access Road",
        description="Width of the road or alley accessing the property in meters (m)"
    )
    floors: float = Field(
        default=3.0,
        ge=1.0,
        le=50.0,
        alias="Floors",
        description="Number of built floors"
    )
    bedrooms: float = Field(
        default=3.0,
        ge=1.0,
        le=30.0,
        alias="Bedrooms",
        description="Number of bedrooms"
    )
    bathrooms: float = Field(
        default=3.0,
        ge=1.0,
        le=30.0,
        alias="Bathrooms",
        description="Number of bathrooms"
    )
    province_city: str = Field(
        default="Hồ Chí Minh",
        alias="Province_City",
        description="Administrative province or city where the property is located"
    )
    legal_status: str = Field(
        default="Have certificate",
        alias="Legal status",
        description="Legal paperwork status (e.g., 'Have certificate', 'Sale contract', 'Other')"
    )

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary matching original model feature column names."""
        return {
            "Area": float(self.area),
            "Frontage": float(self.frontage),
            "Access Road": float(self.access_road),
            "Floors": float(self.floors),
            "Bedrooms": float(self.bedrooms),
            "Bathrooms": float(self.bathrooms),
            "Province_City": str(self.province_city),
            "Legal status": str(self.legal_status),
        }


class ValuationRangeSchema(BaseModel):
    low_estimate_billion: float = Field(description="Lower bound valuation estimate (Billion VND)")
    high_estimate_billion: float = Field(description="Upper bound valuation estimate (Billion VND)")
    confidence_interval_note: str = Field(description="Explanation of empirical RMSE valuation band")


class ValuationResponseSchema(BaseModel):
    predicted_price_billion: float = Field(description="Estimated fair market value in Billion VND")
    formatted_price_vnd: str = Field(description="Formatted Vietnamese currency representation (₫)")
    price_per_m2_million: float = Field(description="Valuation per square meter in Million VND/m²")
    valuation_range: ValuationRangeSchema = Field(description="Estimated price range reflecting model uncertainty")
    engineered_features: Dict[str, float] = Field(
        description="Domain features computed by the transformer (Total_Floor_Area, Bed_Bath_Ratio, Room_Density)"
    )
    champion_model: str = Field(description="Name of the champion model serving inference")
    model_metrics: Dict[str, Any] = Field(description="Historical evaluation metrics of champion model")


class HealthResponseSchema(BaseModel):
    status: str = Field(description="API service health status")
    model_loaded: bool = Field(description="Whether the ML inference pipeline is loaded in memory")
    version: str = Field(description="API version")


class ModelInfoSchema(BaseModel):
    application: str
    champion_model: str
    target: Dict[str, Any]
    input_features: Dict[str, Any]
    engineered_features: List[str]
    transformed_feature_dimension: int
    top_provinces: List[str]
    metrics: Dict[str, Any]
