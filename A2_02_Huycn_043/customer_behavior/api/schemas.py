"""
Pydantic data validation schemas for Customer Behavior REST API.
Adheres to Assignment 02 Appendix C & D specifications.
Supports both standard snake_case and dataset PascalCase aliases.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class CustomerInputSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(
        default=32,
        ge=18,
        le=120,
        alias="Age",
        description="Customer age in years (18-120)"
    )
    rating: int = Field(
        default=5,
        ge=1,
        le=5,
        alias="Rating",
        description="Product rating score (1 to 5 stars)"
    )
    recommended_ind: int = Field(
        default=1,
        ge=0,
        le=1,
        alias="Recommended IND",
        description="Product recommendation indicator (1: Recommended, 0: Not recommended)"
    )
    positive_feedback_count: int = Field(
        default=4,
        ge=0,
        le=1000,
        alias="Positive Feedback Count",
        description="Number of positive upvotes received by this customer review"
    )
    title: Optional[str] = Field(
        default="Stunning summer maxi dress",
        max_length=200,
        alias="Title",
        description="Review headline or summary title"
    )
    review_text: str = Field(
        default="This dress fits like a glove! Beautiful floral fabric and perfect length for weddings.",
        min_length=1,
        max_length=5000,
        alias="Review Text",
        description="Customer detailed review narrative"
    )

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert validated inputs to DataFrame-compatible dictionary matching training columns."""
        return {
            "Age": int(self.age),
            "Rating": int(self.rating),
            "Recommended IND": int(self.recommended_ind),
            "Positive Feedback Count": int(self.positive_feedback_count),
            "Title": str(self.title) if self.title is not None else "",
            "Review Text": str(self.review_text),
        }


class DepartmentProbabilitySchema(BaseModel):
    department: str = Field(description="Department / product category name")
    probability: float = Field(description="Model posterior probability (0.0 to 1.0)")
    percentage: float = Field(description="Probability formatted as percentage (0.0% to 100.0%)")


class PredictionResponseSchema(BaseModel):
    predicted_department: str = Field(
        description="Winning department category prediction (e.g. Dresses, Bottoms, Jackets, Intimate, Tops)"
    )
    confidence: float = Field(
        description="Maximum posterior class probability output by multinomial softmax"
    )
    probabilities: Dict[str, float] = Field(
        description="Dictionary mapping each department to its posterior probability"
    )
    ranked_departments: List[DepartmentProbabilitySchema] = Field(
        description="Sorted list of departments by descending likelihood"
    )
    behavior_category: str = Field(
        description="Triage categorization of customer engagement (e.g., High-Intent Advocate, Critical Reviewer)"
    )
    interpretation: str = Field(
        description="Personalized merchandising and marketing recommendation based on predicted interest"
    )
    engineered_features: Dict[str, Any] = Field(
        description="Multimodal and domain-engineered features extracted from text and engagement metrics"
    )
    champion_model: str = Field(
        description="Model architecture serving real-time inference"
    )


class HealthResponseSchema(BaseModel):
    status: str = Field(description="API service health status")
    model_loaded: bool = Field(description="Whether the multimodal inference pipeline is loaded in memory")
    version: str = Field(description="API version")


class ModelInfoSchema(BaseModel):
    application: str = Field(description="Application title and scope")
    domain: str = Field(description="Business domain")
    champion_model: str = Field(description="Selected champion model name")
    model_family: str = Field(description="Underlying algorithmic family")
    target_classes: List[str] = Field(description="Predictive classes")
    feature_dimensions: Dict[str, int] = Field(description="Dimensionality of text, tabular, and combined vectors")
    metrics: Dict[str, Any] = Field(description="Evaluation metrics from holdout test set")
