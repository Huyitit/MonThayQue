"""
Domain-specific feature engineering transformers for House Price Prediction.
Shared module ensuring seamless serialization/deserialization across Jupyter
notebooks, standalone test scripts, and FastAPI deployment.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERICAL_FEATURES = [
    'Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms'
]

CATEGORICAL_FEATURES = [
    'Province_City', 'Legal status'
]

FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TOP_PROVINCES = [
    'Hồ Chí Minh', 'Hà Nội', 'Bình Dương', 'Đà Nẵng', 'Đồng Nai',
    'Hải Phòng', 'Khánh Hòa', 'Hưng Yên', 'Long An', 'Bà Rịa Vũng Tàu'
]

ENGINEERED_NUMERICAL_FEATURES = [
    'Total_Floor_Area', 'Bed_Bath_Ratio', 'Room_Density'
]

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to compute domain-informed real estate features:
    1. Total_Floor_Area: Area * Floors (proxy for gross usable living space)
    2. Bed_Bath_Ratio: Bedrooms / (Bathrooms + 1.0) (room composition harmony)
    3. Room_Density: (Bedrooms + Bathrooms) / (Area + 1e-5) (room spatial density)
    4. Top Province grouping: Collapses long-tail locations into 'Other'
    
    Accepts both pandas DataFrames and NumPy arrays.
    """
    def __init__(self, add_interactions=True):
        self.add_interactions = add_interactions
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = FEATURE_NAMES.copy()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            cols = self.feature_names_in_ if self.feature_names_in_ else FEATURE_NAMES
            X_df = pd.DataFrame(X, columns=cols[:X.shape[1]])

        if self.add_interactions:
            # Safe extraction with fallback
            area = pd.to_numeric(X_df['Area'], errors='coerce').fillna(50.0)
            floors = pd.to_numeric(X_df['Floors'], errors='coerce').fillna(1.0)
            bedrooms = pd.to_numeric(X_df['Bedrooms'], errors='coerce').fillna(1.0)
            bathrooms = pd.to_numeric(X_df['Bathrooms'], errors='coerce').fillna(1.0)

            # Compute interaction terms
            X_df['Total_Floor_Area'] = area * floors
            X_df['Bed_Bath_Ratio'] = bedrooms / (bathrooms + 1.0)
            X_df['Room_Density'] = (bedrooms + bathrooms) / (area + 1e-5)

        # Standardize province to top 10 + Other
        if 'Province_City' in X_df.columns:
            X_df['Province_City'] = X_df['Province_City'].apply(
                lambda p: p if p in TOP_PROVINCES else 'Other'
            )

        self.feature_names_out_ = list(X_df.columns)
        return X_df
