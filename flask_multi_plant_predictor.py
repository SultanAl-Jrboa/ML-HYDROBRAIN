
from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib
import numpy as np
import pandas as pd

# Load the trained multi-plant model
model = joblib.load("multi_plant_harvest_predictor.pkl")

app = Flask(__name__)

# Enable CORS for all routes with specific origins
CORS(app, resources={
    r"/predict": {
        "origins": [
            "http://127.0.0.1:5500",  
            "http://localhost:5500",   
            "https://hydrobrain.ngrok.dev",  
            "https://hydrobrain.sultanaljarboa.sa"  
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "Origin"]
    }
})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame with expected input structure
        input_df = pd.DataFrame([{
            "Plant_Type": data["plant_type"],
            "Day_Since_Planting": data["day_since_planting"],
            "Temperature_C": data["temperature"],
            "pH": data["pH"],
            "EC_dS_per_m": data["ec"],
            "Light_Hours": data["light_hours"],
            "Humidity_Percent": data["humidity"]
        }])

        prediction = model.predict(input_df)
        return jsonify({"predicted_days_to_harvest": round(prediction[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
