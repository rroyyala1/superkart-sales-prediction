# Import the libraries
import os

import pandas as pd
import requests
import streamlit as st

# Base URL of the Flask backend
# - inside the Docker network the backend container is reachable by its container name ("backend")
# - outside Docker, set BACKEND_URL=http://localhost:7860
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

# Page title
st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒")
st.title("SuperKart Sales Prediction")
st.write("Enter the product and store details below to predict the total sales of the product in the store.")

# ---------------------------------------------------------------- Online prediction
st.subheader("Online Prediction")

# Input fields for product and store data (only the categories the model was trained on)
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1)
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"], help="FD = Food, DR = Drinks, NC = Non-consumables")
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"], index=1)

# Create JSON payload in the same format the backend expects
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": int(Store_Age_Years),
    "Product_Type_Category": Product_Type_Category,
}

# Send the payload to the backend when the user clicks Predict
if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            prediction = response.json()["Predicted_Product_Store_Sales_Total"]
            st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
        else:
            st.error(f"Error from the API ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend at {BACKEND_URL}: {e}")

# ---------------------------------------------------------------- Batch prediction
st.subheader("Batch Prediction")
st.write("Upload a CSV file with the columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, "
         "Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, "
         "Store_Age_Years, Product_Type_Category")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if st.button("Predict Batch"):
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
    else:
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                     files={"file": uploaded_file.getvalue()}, timeout=120)
            if response.status_code == 200:
                predictions = response.json()
                # Attach the predictions to the uploaded rows and show them
                results = pd.read_csv(uploaded_file)
                results["Predicted_Sales"] = [predictions[str(i)] for i in results.index]
                st.success(f"Batch predictions completed for {len(results)} rows.")
                st.dataframe(results)
                # Let the user download the predictions
                st.download_button("Download Predictions", results.to_csv(index=False),
                                   file_name="superkart_predictions.csv", mime="text/csv")
            else:
                st.error(f"Error from the API ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend at {BACKEND_URL}: {e}")
