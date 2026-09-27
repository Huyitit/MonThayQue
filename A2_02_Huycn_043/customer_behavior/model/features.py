"""
Domain-specific feature engineering transformers for E-Commerce Customer Behavior.
Shared module ensuring seamless serialization/deserialization across Jupyter
notebooks, standalone test scripts, and FastAPI deployment.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_NUMERICAL_FEATURES = [
    'Age', 'Rating', 'Recommended IND', 'Positive Feedback Count'
]

ENGINEERED_NUMERICAL_FEATURES = [
    'review_length', 'word_count', 'log_positive_feedback',
    'uppercase_ratio', 'has_title'
]

ALL_TABULAR_FEATURES = [
    'Age', 'Rating', 'Recommended IND', 'log_positive_feedback',
    'word_count', 'review_length'
]

TEXT_FEATURE = 'clean_text'

TARGET_COLUMN = 'Department Name'

TARGET_CLASSES = ['Tops', 'Dresses', 'Bottoms', 'Intimate', 'Jackets']


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to engineer customer behavior features:
    1. review_length: Total character count of review text
    2. word_count: Number of tokens/words in review text
    3. log_positive_feedback: Log1p transform of Positive Feedback Count
    4. uppercase_ratio: Proportion of uppercase characters in text
    5. has_title: Binary flag indicating if title was provided
    6. clean_text: Concatenated Title + Review Text stream for NLP vectorizer
    
    Accepts pandas DataFrames and returns an enriched DataFrame.
    """
    def __init__(self, add_text_metrics=True):
        self.add_text_metrics = add_text_metrics
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            cols = self.feature_names_in_ if self.feature_names_in_ else RAW_NUMERICAL_FEATURES
            X_df = pd.DataFrame(X, columns=cols[:X.shape[1]])

        # Ensure text fields exist and are string
        if 'Review Text' in X_df.columns:
            review_text = X_df['Review Text'].fillna('').astype(str).str.strip()
        else:
            review_text = pd.Series([''] * len(X_df), index=X_df.index)

        if 'Title' in X_df.columns:
            title_text = X_df['Title'].fillna('').astype(str).str.strip()
        else:
            title_text = pd.Series([''] * len(X_df), index=X_df.index)

        # Construct clean_text if not already present
        if 'clean_text' not in X_df.columns:
            has_title_mask = title_text != ''
            clean_text = np.where(
                has_title_mask,
                title_text + ". " + review_text,
                review_text
            )
            X_df['clean_text'] = clean_text

        if self.add_text_metrics:
            # 1. review_length
            X_df['review_length'] = review_text.str.len()
            
            # 2. word_count
            X_df['word_count'] = review_text.apply(lambda s: len(s.split()))
            
            # 3. log_positive_feedback
            if 'Positive Feedback Count' in X_df.columns:
                feedback = pd.to_numeric(X_df['Positive Feedback Count'], errors='coerce').fillna(0.0)
                X_df['log_positive_feedback'] = np.log1p(np.maximum(feedback, 0.0))
            else:
                X_df['log_positive_feedback'] = 0.0

            # 4. uppercase_ratio
            def calc_upper_ratio(s):
                if len(s) == 0:
                    return 0.0
                return sum(1 for c in s if c.isupper()) / max(len(s), 1)

            X_df['uppercase_ratio'] = review_text.apply(calc_upper_ratio)

            # 5. has_title
            X_df['has_title'] = (title_text != '').astype(int)

        self.feature_names_out_ = list(X_df.columns)
        return X_df
