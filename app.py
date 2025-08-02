import pickle

import numpy as np

from flask import Flask, request, jsonify
 
# Load the trained model and scaler

model = pickle.load(open("pcos_rf_model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))
 
# List of feature names (must match the training data)

feature_names = ['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)', 

                 'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)', 

                 'Pregnant(Y/N)', 'No. of abortions', 'I beta-HCG(mIU/mL)', 'II beta-HCG(mIU/mL)', 

                 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio', 

                 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 

                 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 

                 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 

                 'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)', 

                 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)']
 
# Initialize Flask app

app = Flask(__name__)
 
@app.route('/predict', methods=['POST'])

def predict():

    try:

        data = request.json

        features = data.get('features')
 
        if not isinstance(features, list) or len(features) != len(feature_names):

            return jsonify({'error': f'Expected {len(feature_names)} features'}), 400

        if not all(isinstance(x, (int, float)) for x in features):

            return jsonify({'error': 'All features must be numbers'}), 400
 
        features_array = np.array(features).reshape(1, -1)

        features_scaled = scaler.transform(features_array)

        prediction = int(model.predict(features_scaled)[0])

        try:

            prob = model.predict_proba(features_scaled)[0][1]

        except:

            prob = None
 
        return jsonify({

            'prediction': prediction,

            'message': 'PCOS Detected' if prediction == 1 else 'No PCOS Detected',

            'probability': f"{round(prob * 100, 2)}%" if prob is not None else 'Not Available',

            'features': {feature_names[i]: features[i] for i in range(len(features))}

        })

    except Exception as e:

        return jsonify({'error': str(e)}), 500
 
# Entry point (optional, not used in Render)

if __name__ == '__main__':

    app.run(debug=False)

 