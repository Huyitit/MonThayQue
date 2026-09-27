"""
Domain-specific feature engineering transformers for Diabetes Prediction.
Shared module to ensure seamless serialization/deserialization across Jupyter notebooks,
standalone test scripts, and FastAPI deployment.
"""
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

ENGINEERED_FEATURE_NAMES = FEATURE_NAMES + [
    'Glucose_BMI_Interaction',
    'Insulin_Glucose_Ratio'
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to compute domain-informed clinical features:
    1. Glucose_BMI_Interaction: Glucose * BMI (metabolic risk synergy)
    2. Insulin_Glucose_Ratio: Insulin / (Glucose + 1e-5) (pancreatic compensation proxy)
    
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
            glucose = X_df['Glucose'] if 'Glucose' in X_df.columns else X_df.iloc[:, 1]
            bmi = X_df['BMI'] if 'BMI' in X_df.columns else X_df.iloc[:, 5]
            insulin = X_df['Insulin'] if 'Insulin' in X_df.columns else X_df.iloc[:, 4]

            X_df['Glucose_BMI_Interaction'] = glucose * bmi
            X_df['Insulin_Glucose_Ratio'] = insulin / (glucose + 1e-5)

        self.feature_names_out_ = list(X_df.columns)
        return X_df
