# predict.py

import joblib
import pandas as pd
from Custom_Transformers import CombinedAttributesAdder

# Load model
model = joblib.load("housing_model.pkl")

input_data = {
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [890.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
}

input_df = pd.DataFrame(input_data)

# Make prediction
predicted_value = model.predict(input_df)[0]
print(f"🏡 Predicted Median House Value: ${predicted_value:,.2f}")
