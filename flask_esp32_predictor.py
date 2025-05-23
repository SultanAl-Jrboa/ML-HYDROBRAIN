# === Step 1: flask_esp32_predictor.py ===

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("lettuce_harvest_predictor.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    data = request.get_json()
    try:
        input_features = [
            data['day_since_planting'],
            data['temperature'],
            data['pH'],
            data['ec'],
            data['light_hours'],
            data['humidity']
        ]
        prediction = model.predict([input_features])
        return jsonify({'predicted_days_to_harvest': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
