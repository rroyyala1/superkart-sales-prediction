# Import the libraries
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
superkart_api = Flask("SuperKart Sales Predictor")

# Load the serialized pipeline (preprocessing + tuned model) once at start-up
model = joblib.load("superkart_model.joblib")

# Features expected by the model - this is the input contract of the API
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL_FEATURES = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
                        "Store_Type", "Product_Id_char", "Product_Type_Category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df):
    """Check that all features are present, put them in the model's order and fix the data types."""
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    df = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")  # raises an error for non-numeric values
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()  # remove stray spaces around category labels
    return df


# Home / health-check route
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


# Online prediction: one product-store record sent as JSON
@superkart_api.post("/v1/predict")
def predict_sales():
    # Get the JSON data from the request body
    sales_data = request.get_json(silent=True)
    if not isinstance(sales_data, dict):
        return jsonify({"error": "Request body must be a JSON object with the model features."}), 400

    # Convert the JSON into a one-row DataFrame and validate it
    try:
        input_data = prepare_features(pd.DataFrame([sales_data]))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    # Predict the sales and return it rounded to 2 decimals
    prediction = model.predict(input_data)[0]
    return jsonify({"Predicted_Product_Store_Sales_Total": round(float(prediction), 2)})


# Batch prediction: a CSV file with many product-store records
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Upload a CSV file in the 'file' form field."}), 400

    # Read the CSV into a DataFrame and validate it
    try:
        input_data = prepare_features(pd.read_csv(file))
    except Exception as e:
        return jsonify({"error": f"Invalid input file: {e}"}), 400

    # Predict for all rows and return {row_index: predicted_sales}
    predictions = model.predict(input_data)
    output = {str(idx): round(float(pred), 2) for idx, pred in zip(input_data.index, predictions)}
    return jsonify(output)


# Run the app locally (inside Docker the app is started by gunicorn instead)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
