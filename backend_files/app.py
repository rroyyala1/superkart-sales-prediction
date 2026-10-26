# Import the libraries
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
superkart_api = Flask("SuperKart Sales Predictor")

# Version of the API contract - the notebook checks it to make sure it talks to the latest deployment
API_VERSION = "1.1"

# Load the serialized pipeline (preprocessing + tuned model) once at start-up
model = joblib.load("superkart_model.joblib")

# Features expected by the model - this is the input contract of the API
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL_FEATURES = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
                        "Store_Type", "Product_Id_char", "Product_Type_Category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valid range for each numeric feature: (minimum, maximum) - None means no upper limit
NUMERIC_RANGES = {
    "Product_Weight": (0, None),          # weight cannot be negative
    "Product_Allocated_Area": (0, 1),     # a ratio of the total display area
    "Product_MRP": (0, None),             # price cannot be negative
    "Store_Age_Years": (0, None),         # age cannot be negative
}

# Categories seen in training, read from the fitted one-hot encoder inside the pipeline
_encoder = model[0].named_transformers_["onehotencoder"]
KNOWN_CATEGORIES = {col: set(cats) for col, cats in zip(_encoder.feature_names_in_, _encoder.categories_)}


def prepare_features(df):
    """
    Validate the input rows and return (clean DataFrame, warnings).
    Raises ValueError with a readable message when the input cannot be scored.
    """
    if df.empty:
        raise ValueError("No rows to predict - the input is empty.")

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    df = df[FEATURES].copy()
    errors = []

    # Numeric features: must be real, finite numbers inside their valid range
    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")  # text such as "abc" becomes NaN
        bad = values.isna() | ~np.isfinite(values.astype(float))  # nulls, text, inf and -inf
        if bad.any():
            errors.append(f"{col}: missing, non-numeric or infinite value in row(s) {list(df.index[bad])[:5]}")
        low, high = NUMERIC_RANGES[col]
        out_of_range = (~bad) & ((values < low) | ((values > high) if high is not None else False))
        if out_of_range.any():
            limit = f">= {low}" if high is None else f"between {low} and {high}"
            errors.append(f"{col}: must be {limit} - check row(s) {list(df.index[out_of_range])[:5]}")
        df[col] = values

    # Categorical features: must be present and non-blank
    for col in CATEGORICAL_FEATURES:
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if blank.any():
            errors.append(f"{col}: missing value in row(s) {list(df.index[blank])[:5]}")
        df[col] = df[col].astype(str).str.strip()  # remove stray spaces around category labels

    if errors:
        raise ValueError("; ".join(errors))

    # Categories never seen in training are still scored (encoded as all zeros),
    # but the caller is told that these predictions are extrapolations
    warnings = []
    for col in CATEGORICAL_FEATURES:
        unseen = sorted(set(df[col]) - KNOWN_CATEGORIES[col])
        if unseen:
            warnings.append(f"{col}: {unseen} not seen in training - prediction is an extrapolation")
    return df, warnings


# Home route
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


# Health route - lets the notebook / monitoring confirm which API version is deployed
@superkart_api.get("/v1/health")
def health():
    return jsonify({"status": "ok", "api_version": API_VERSION})


# Online prediction: one product-store record sent as JSON
@superkart_api.post("/v1/predict")
def predict_sales():
    # Get the JSON data from the request body
    sales_data = request.get_json(silent=True)
    if not isinstance(sales_data, dict):
        return jsonify({"error": "Request body must be a JSON object with the model features."}), 400

    # Convert the JSON into a one-row DataFrame and validate it
    try:
        input_data, warnings = prepare_features(pd.DataFrame([sales_data]))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    # Predict - any unexpected failure returns a JSON error instead of an HTML 500 page
    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Return the prediction rounded to 2 decimals, plus any warnings
    return jsonify({"Predicted_Product_Store_Sales_Total": round(prediction, 2), "warnings": warnings})


# Batch prediction: a CSV file with many product-store records
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Upload a CSV file in the 'file' form field."}), 400

    # Read the CSV into a DataFrame and validate it (an empty or unreadable file is a 400, not a crash)
    try:
        input_data, warnings = prepare_features(pd.read_csv(file))
    except pd.errors.EmptyDataError:
        return jsonify({"error": "Invalid input file: the CSV file is empty."}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid input file: {e}"}), 400

    # Predict for all rows - any unexpected failure returns a JSON error
    try:
        predictions = model.predict(input_data)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Body keeps the original contract {row_index: predicted_sales};
    # warnings about unseen categories travel in a response header
    output = {str(idx): round(float(pred), 2) for idx, pred in zip(input_data.index, predictions)}
    response = jsonify(output)
    if warnings:
        response.headers["X-Prediction-Warnings"] = " | ".join(warnings)
    return response


# Run the app locally (inside Docker the app is started by gunicorn instead)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
