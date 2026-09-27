import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_address(addr):
    if not isinstance(addr, str):
        return "Unknown", "Unknown"
    parts = [p.strip() for p in addr.split(',')]
    province = parts[-1] if len(parts) > 0 else "Unknown"
    district = parts[-2] if len(parts) > 1 else "Unknown"
    return province, district

def export_pipeline_and_model():
    print("Loading dataset...")
    csv_path = os.path.join(os.path.dirname(__file__), "../dataset/vn_house.dataset.csv")
    df = pd.read_csv(csv_path)
    
    # Pre-split processing: address parsing
    parsed = df['Address'].apply(parse_address)
    df['City_Province'] = [p[0] for p in parsed]
    df['District'] = [p[1] for p in parsed]
    df = df.drop(columns=['Address'])
    
    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    X_train = train_df.drop(columns=['Price']).copy()
    X_test = test_df.drop(columns=['Price']).copy()
    y_train = np.log1p(train_df['Price'])
    y_test = np.log1p(test_df['Price'])
    
    # Impute numeric features
    num_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
    medians = {}
    for col in num_cols:
        median_val = X_train[col].median()
        medians[col] = median_val
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
        
    # Impute categorical features
    cat_cols = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Unknown")
        X_test[col] = X_test[col].fillna("Unknown")
        
    # Target Encoding for location columns
    te_cols = ['City_Province', 'District']
    te = TargetEncoder(cv=5)
    X_train_te = te.fit_transform(X_train[te_cols], y_train)
    X_test_te = te.transform(X_test[te_cols])
    
    X_train_te_df = pd.DataFrame(X_train_te, columns=[f"{col}_encoded" for col in te_cols], index=X_train.index)
    X_test_te_df = pd.DataFrame(X_test_te, columns=[f"{col}_encoded" for col in te_cols], index=X_test.index)
    
    X_train = pd.concat([X_train.drop(columns=te_cols), X_train_te_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=te_cols), X_test_te_df], axis=1)
    
    # One-Hot Encoding
    X_train = pd.get_dummies(X_train, columns=cat_cols)
    X_test = pd.get_dummies(X_test, columns=cat_cols)
    
    # Column Alignment
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Keep track of columns
    X_train_columns = list(X_train.columns)
    
    print("Fitting Random Forest Pipeline...")
    # Pipeline components
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols)
        ],
        remainder='passthrough'
    )
    
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate and verify against notebook values
    y_pred_test_log = rf_pipeline.predict(X_test)
    y_test_orig = np.expm1(y_test)
    y_pred_test_orig = np.expm1(y_pred_test_log)
    
    test_r2_log = r2_score(y_test, y_pred_test_log)
    test_mae_orig = mean_absolute_error(y_test_orig, y_pred_test_orig)
    test_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    
    print("\n--- Validation Results ---")
    print(f"Test R2 (Log Scale): {test_r2_log:.6f} (Expected: ~0.607398)")
    print(f"Test MAE (Original Scale - Billion VND): {test_mae_orig:.6f} (Expected: ~1.114250)")
    print(f"Test RMSE (Original Scale - Billion VND): {test_rmse_orig:.6f} (Expected: ~1.463078)")
    
    # Save artifacts
    model_dir = os.path.dirname(__file__)
    te_path = os.path.join(model_dir, "te.joblib")
    rf_pipeline_path = os.path.join(model_dir, "rf_pipeline.joblib")
    metadata_path = os.path.join(model_dir, "preprocessing_metadata.joblib")
    
    print(f"\nSaving fitted target encoder to: {te_path}")
    joblib.dump(te, te_path)
    
    print(f"Saving fitted RF pipeline to: {rf_pipeline_path}")
    joblib.dump(rf_pipeline, rf_pipeline_path)
    
    metadata = {
        "medians": medians,
        "cat_cols": cat_cols,
        "te_cols": te_cols,
        "X_train_columns": X_train_columns,
        "num_cols": num_cols
    }
    print(f"Saving preprocessing metadata to: {metadata_path}")
    joblib.dump(metadata, metadata_path)
    print("Done!")

if __name__ == "__main__":
    export_pipeline_and_model()
