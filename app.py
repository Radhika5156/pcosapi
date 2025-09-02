# app.py
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Load models & scaler (ensure these files exist in the repo) ---
with open("pcos_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("pcos_svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Feature names (order must match how scaler/model were trained) ---
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

# Map friendly keys to indices in feature vector
key_to_index = {
    'age': 0, 'weight': 1, 'height': 2, 'bmi': 3, 'blood_group': 4,
    'pulse': 5, 'rr': 6, 'hb': 7, 'cycle': 8, 'cycle_length': 9,
    'marriage_status': 10, 'pregnant': 11, 'no_of_abortions': 12,
    'i_beta_hcg': 13, 'ii_beta_hcg': 14, 'fsh': 15, 'lh': 16, 'fsh_lh': 17,
    'hip': 18, 'waist': 19, 'waist_hip_ratio': 20, 'tsh': 21, 'amh': 22,
    'prl': 23, 'vitd3': 24, 'prg': 25, 'rbs': 26, 'weight_gain': 27,
    'hair_growth': 28, 'skin_darkening': 29, 'hair_loss': 30, 'pimples': 31,
    'fast_food': 32, 'reg_exercise': 33, 'bp_systolic': 34, 'bp_diastolic': 35,
    'follicle_l': 36, 'follicle_r': 37, 'avg_f_size_l': 38, 'avg_f_size_r': 39,
    'endometrium': 40
}

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (useful for mobile/web clients)


def to_float_safe(v, default=0.0):
    """Try to convert to float; handle None / empty / strings like 'NA'."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return default
    try:
        return float(s)
    except:
        # For Y/N or boolean-ish strings return default (0.0) here
        return default


def yn_to_int(v):
    """Convert Y/N/Yes/No/1/0/True/False (case-insensitive) to 1 or 0."""
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return 1 if v != 0 else 0
    s = str(v).strip().lower()
    if s in ("y", "yes", "true", "1", "t", "y(es)"):
        return 1
    if s in ("n", "no", "false", "0"):
        return 0
    # special: for gender 'f' -> 1 (we keep previous behavior)
    if s in ("f", "female"):
        return 1
    return 0


def safe_string(val):
    if val is None:
        return ""
    return str(val).strip()


def build_feature_vector_from_input(data):
    """
    Accepts:
      - data["features"] as array of length 41 (preferred)
      - or data["features"] as short array of length 5: [age, bmi, cycle_length, bp, gender]
      - or a JSON object with named keys (see key_to_index)
    Returns: list of 41 floats (ready to scale)
    """
    # start with zeros
    vec = [0.0] * len(feature_names)

    # 1) If features array provided
    if "features" in data:
        features_in = data["features"]
        if isinstance(features_in, list) or isinstance(features_in, tuple):
            n = len(features_in)
            if n == len(feature_names):
                # convert each to numeric where possible, handle Y/N -> 0/1
                for i, val in enumerate(features_in):
                    # treat symptom columns (indices 27..33, 11 maybe pregnant) as yn ints
                    if i in (11, 27, 28, 29, 30, 31, 32, 33):
                        vec[i] = float(yn_to_int(val))
                    else:
                        # try numeric conversion otherwise 0
                        try:
                            vec[i] = float(val)
                        except:
                            vec[i] = 0.0
                return vec
            elif n == 5:
                # backward compatibility: interpret 5 values as [age, bmi, cycle_length, bp, gender]
                age, bmi, cycle_len, bp, gender = features_in
                vec[key_to_index['age']] = to_float_safe(age)
                vec[key_to_index['bmi']] = to_float_safe(bmi)
                vec[key_to_index['cycle_length']] = to_float_safe(cycle_len)
                # map systolic BP to bp_systolic; put same value in diastolic if desired
                vec[key_to_index['bp_systolic']] = to_float_safe(bp)
                vec[key_to_index['bp_diastolic']] = to_float_safe(bp)
                # previous code used gender -> index 4 for compatibility (Blood Group field was reused)
                vec[4] = float(1 if str(gender).strip().upper() == "F" else 0)
                return vec
            else:
                # unexpected length: we'll fall through to try named keys
                pass

    # 2) Named keys: fill from key_to_index mapping
    for k, idx in key_to_index.items():
        if k in data:
            v = data[k]
            # symptoms & boolean-ish fields
            if k in ("pregnant", "weight_gain", "hair_growth", "skin_darkening",
                     "hair_loss", "pimples", "fast_food", "reg_exercise"):
                vec[idx] = float(yn_to_int(v))
            else:
                # try to cast to float (numeric)
                vec[idx] = to_float_safe(v)

    # 3) Some special handling: accept a few common aliases
    # e.g., "cycle_length" or "cycleLen", "age", "bmi", "bp"
    if "age" in data and vec[key_to_index['age']] == 0:
        vec[key_to_index['age']] = to_float_safe(data.get("age"))
    if "bmi" in data and vec[key_to_index['bmi']] == 0:
        vec[key_to_index['bmi']] = to_float_safe(data.get("bmi"))
    # if gender provided as "F"/"M" outside above mapping, put it into index 4 for backward compat
    if "gender" in data and (vec[4] == 0):
        vec[4] = float(1 if str(data.get("gender")).strip().upper() == "F" else 0)

    return vec


def generate_advice(prediction, provided_data):
    """
    Simple rule-based advice generator. Returns a list of advice strings.
    provided_data: the original JSON payload (useful to personalize).
    """
    adv = []

    # general tips
    if prediction == 1:
        adv.append("Likely PCOS detected. Please consult a gynecologist/endocrinologist for confirmation.")
        adv.append("Diet: prefer low-glycemic, high-fiber foods; reduce sugary and refined carbs.")
        adv.append("Exercise: aim for 30 minutes daily (walk/yoga/strength training).")
        adv.append("Sleep & stress: minimize screen before bed, aim 7-8 hours.")
    else:
        adv.append("Low likelihood of PCOS based on the provided inputs.")
        adv.append("Maintain balanced diet and regular activity to keep symptoms low.")

    # targeted tips based on symptoms flags
    # weight gain
    if provided_data.get("weight_gain") in ("Y", "y", "yes", 1, True) or str(provided_data.get("weight_gain")).lower() == "1":
        adv.append("Weight management: small sustained calorie deficit + resistance training.")
    # hair growth / hirsutism
    if provided_data.get("hair_growth") in ("Y", "y", "yes", 1, True) or str(provided_data.get("hair_growth")).lower() == "1":
        adv.append("Hair growth: discuss with doctor; topical/medical options and lifestyle changes can help.")
    # acne/pimples
    if provided_data.get("pimples") in ("Y", "y", "yes", 1, True) or str(provided_data.get("pimples")).lower() == "1":
        adv.append("Acne: see a dermatologist; reduce dairy & high glycemic foods; consider topical treatments.")
    # irregular cycle
    cycle_val = provided_data.get("cycle") or provided_data.get("cycle_status") or provided_data.get("period_regular")
    if cycle_val:
        s = str(cycle_val).strip().lower()
        if s in ("i", "irregular", "irregular_period", "no", "n", "irregular"):
            adv.append("Irregular cycles: track cycles and consult doctor for hormonal tests.")
    # vitamin D / sleep advice
    if provided_data.get("vitd3"):
        try:
            if float(provided_data.get("vitd3")) < 20:
                adv.append("Low Vitamin D detected: consider testing and supplementing under medical advice.")
        except:
            pass

    return adv


# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return "PCOS Prediction API is running. Use /predict endpoint with POST request."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True) or {}
        print("INCOMING PAYLOAD:", data)

        model_choice = str(data.get("model", "rf")).lower()
        if model_choice not in ("rf", "svm"):
            model_choice = "rf"

        # Build 41-length feature vector
        features = build_feature_vector_from_input(data)

        # Defensive check
        if len(features) != len(feature_names):
            return jsonify({"error": f"Feature vector length {len(features)} incorrect, expected {len(feature_names)}."}), 400

        # scale and predict
        features_array = np.array(features).reshape(1, -1)
        try:
            features_scaled = scaler.transform(features_array)
        except Exception as e:
            return jsonify({"error": f"Scaler error: {str(e)}"}), 500

        model = rf_model if model_choice == "rf" else svm_model

        try:
            pred_raw = model.predict(features_scaled)[0]
            prediction = int(pred_raw)
        except Exception as e:
            return jsonify({"error": f"Model predict error: {str(e)}"}), 500

        # probability if available
        prob = None
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(features_scaled)[0][1]
        except Exception:
            prob = None

        advice = generate_advice(prediction, data)

        response = {
            "model_used": model_choice,
            "prediction": prediction,
            "message": "PCOS Detected" if prediction == 1 else "No PCOS Detected",
            "probability": f"{round(prob * 100, 2)}%" if prob is not None else "Not Available",
            "features_used": {k: features[key_to_index[k]] if k in key_to_index else None for k in key_to_index},
            "advice": advice
        }

        return jsonify(response)

    except Exception as e:
        # unexpected
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False for production
    app.run(host="0.0.0.0", port=port, debug=False)
