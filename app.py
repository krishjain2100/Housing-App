# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
from Custom_Transformers import CombinedAttributesAdder

# Load model pipeline
print("Loading model...")
model = joblib.load("housing_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'longitude': [float(request.form['longitude'])],
            'latitude': [float(request.form['latitude'])],
            'housing_median_age': [float(request.form['housing_median_age'])],
            'total_rooms': [float(request.form['total_rooms'])],
            'total_bedrooms': [float(request.form['total_bedrooms'])],
            'population': [float(request.form['population'])],
            'households': [float(request.form['households'])],
            'median_income': [float(request.form['median_income'])],
            'ocean_proximity': [request.form['ocean_proximity']]
        }

        input_df = pd.DataFrame(input_data)
        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction_text=f"🏡 Predicted House Value: ${prediction:,.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render
    app.run(host='0.0.0.0', port=port, debug=True)

