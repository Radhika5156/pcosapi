import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load models
rf_model = pickle.load(open("pcos_rf_model.pkl", "rb"))
svm_model = pickle.load(open("pcos_svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/', methods=['GET'])
def home():
    return "PCOS Prediction API is running. Use /predict endpoint with POST request."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_choice = data.get('model', 'rf')  # default: Random Forest

        # ✅ Accept both formats: features[] OR separate keys
        if "features" in data:
            features_input = data["features"]
            if len(features_input) != 5:
                return jsonify({"error": "features must have 5 values: [age, bmi, cycle_length, bp, gender]"}), 400
            age, bmi, cycle_len, bp, gender = features_input
            gender = 1 if str(gender).upper() == "F" else 0
        else:
            age = float(data.get("age", 0))
            bmi = float(data.get("bmi", 0))
            cycle_len = float(data.get("cycle_length", 0))
            bp = float(data.get("bp", 0))
            gender = 1 if data.get("gender", "F").upper() == "F" else 0

        # ✅ Fill missing 37 features with zeros
        features = [age, 0, 0, bmi, gender, 0, 0, 0, 0, cycle_len,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, bp, 0, 0, 0, 0, 0, 0]

        # Transform input
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Select model
        if model_choice == 'rf':
            model = rf_model
        elif model_choice == 'svm':
            model = svm_model
        else:
            return jsonify({'error': 'Invalid model. Use "rf" or "svm".'}), 400

        # Predict
        prediction = int(model.predict(features_scaled)[0])
        try:
            prob = model.predict_proba(features_scaled)[0][1]
        except AttributeError:
            prob = None

        return jsonify({
            'model_used': model_choice,
            'prediction': prediction,
            'message': 'PCOS Detected' if prediction == 1 else 'No PCOS Detected',
            'probability': f"{round(prob * 100, 2)}%" if prob is not None else 'Not Available',
            'features_used': {
                "age": age,
                "bmi": bmi,
                "cycle_length": cycle_len,
                "bp": bp,
                "gender": gender
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
