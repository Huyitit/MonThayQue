import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Paths to joblib files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TE_PATH = os.path.join(MODEL_DIR, "te.joblib")
RF_PIPELINE_PATH = os.path.join(MODEL_DIR, "rf_pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "preprocessing_metadata.joblib")

# Load model and preprocessing components
print("Loading serialized model and preprocessing metadata...")
try:
    te = joblib.load(TE_PATH)
    rf_pipeline = joblib.load(RF_PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

def parse_address(addr):
    if not isinstance(addr, str) or not addr.strip():
        return "Unknown", "Unknown"
    parts = [p.strip() for p in addr.split(',')]
    province = parts[-1] if len(parts) > 0 else "Unknown"
    district = parts[-2] if len(parts) > 1 else "Unknown"
    return province, district

@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory(os.path.join(BASE_DIR, "static"), path)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # 1. Input: Extract values from JSON request
        address = data.get("Address", "")
        area = data.get("Area")
        frontage = data.get("Frontage")
        access_road = data.get("Access Road")
        floors = data.get("Floors")
        bedrooms = data.get("Bedrooms")
        bathrooms = data.get("Bathrooms")
        house_dir = data.get("House direction", "Unknown")
        balcony_dir = data.get("Balcony direction", "Unknown")
        legal_status = data.get("Legal status", "Unknown")
        furniture_state = data.get("Furniture state", "Unknown")
        
        # 2. Representation: Address parsing to City_Province and District
        parsed_province, parsed_district = parse_address(address)
        
        # If user directly specified province and district, use those as fallback
        city_province = data.get("City_Province", parsed_province)
        district = data.get("District", parsed_district)
        
        # Create a single-row DataFrame
        row_dict = {
            "Area": [area],
            "Frontage": [frontage],
            "Access Road": [access_road],
            "Floors": [floors],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "House direction": [house_dir],
            "Balcony direction": [balcony_dir],
            "Legal status": [legal_status],
            "Furniture state": [furniture_state],
            "City_Province": [city_province],
            "District": [district]
        }
        df = pd.DataFrame(row_dict)
        
        # 3. Preprocessing:
        # Impute numeric features with medians
        for col in metadata["num_cols"]:
            val = df.loc[0, col]
            # Convert empty strings or None to median
            if val is None or val == "" or pd.isna(val):
                df.loc[0, col] = metadata["medians"][col]
            else:
                df.loc[0, col] = float(val)
                
        # Impute categorical features
        for col in metadata["cat_cols"]:
            val = df.loc[0, col]
            if val is None or val == "" or pd.isna(val):
                df.loc[0, col] = "Unknown"
            else:
                df.loc[0, col] = str(val)
                
        # Target Encoding
        te_cols = metadata["te_cols"]
        df_te = te.transform(df[te_cols])
        
        df_te_df = pd.DataFrame(
            df_te, 
            columns=[f"{col}_encoded" for col in te_cols], 
            index=df.index
        )
        
        df = pd.concat([df.drop(columns=te_cols), df_te_df], axis=1)
        
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=metadata["cat_cols"])
        
        # Column Alignment: reindex to match the training features
        df = df.reindex(columns=metadata["X_train_columns"], fill_value=0)
        
        # 4. Model Prediction
        y_pred_log = rf_pipeline.predict(df)
        
        # 5. Output: Inverse log-transform to Billion VND
        y_pred_orig = np.expm1(y_pred_log[0])
        
        # Gather info about key preprocessing stages for frontend visualizer
        preprocessing_summary = {
            "parsed_location": {
                "City_Province": city_province,
                "District": district
            },
            "target_encoded_location": {
                "City_Province_encoded": float(df_te_df.loc[0, "City_Province_encoded"]),
                "District_encoded": float(df_te_df.loc[0, "District_encoded"])
            },
            "aligned_features_count": len(metadata["X_train_columns"])
        }
        
        return jsonify({
            "price_billion_vnd": float(round(y_pred_orig, 3)),
            "price_vnd": int(round(y_pred_orig * 1e9)),
            "predicted_log": float(round(y_pred_log[0], 6)),
            "preprocessing": preprocessing_summary
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
