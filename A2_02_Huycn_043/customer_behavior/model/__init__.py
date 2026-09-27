"""
Customer Behavior Model Package.
Contains domain feature engineering transformers and serialized model artifacts.
"""
import sys
from . import features
from .features import (
    CustomerFeatureEngineer,
    RAW_NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    ALL_TABULAR_FEATURES,
    TEXT_FEATURE,
    TARGET_COLUMN,
    TARGET_CLASSES,
)

# Register in sys.modules and __main__ to ensure seamless joblib unpickling
if "features" not in sys.modules:
    sys.modules["features"] = features
setattr(sys.modules.get("__main__"), "CustomerFeatureEngineer", CustomerFeatureEngineer)

__all__ = [
    "CustomerFeatureEngineer",
    "RAW_NUMERICAL_FEATURES",
    "ENGINEERED_NUMERICAL_FEATURES",
    "ALL_TABULAR_FEATURES",
    "TEXT_FEATURE",
    "TARGET_COLUMN",
    "TARGET_CLASSES",
]
