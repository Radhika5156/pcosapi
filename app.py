import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load models
rf_model = pickle.load(open("pcos_rf_model.pkl", "rb"))
svm_model = pickle.load(open("pcos_svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feature names (same as training)
feature_names = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)',
    'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)',
    'Pregnant(Y/N)', 'No. of abortions', 'I beta-HCG(mIU/mL)', 'II beta-HCG(mIU/mL)',
    'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio',
    'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)',
    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
    'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)',
    'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
    'Endometrium (mm)'
]

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
        features = data.get('features')
        model_choice = data.get('model', 'rf')  # default: Random Forest

        # Input validation
        if not isinstance(features, list) or len(features) != len(feature_names):
            return jsonify({'error': f'Expected {len(feature_names)} features'}), 400
        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({'error': 'All features must be numbers'}), 400

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
            'features': {feature_names[i]: features[i] for i in range(len(features))}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
