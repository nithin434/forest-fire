from flask import Flask, request, render_template
import json
import numpy as np
import joblib

app = Flask(__name__)

MODEL_PATH = "forestfiremodel.pkl"
FEATURE_ORDER = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]


def load_model():
    return joblib.load(MODEL_PATH)


def load_metrics():
    try:
        with open("outputs/metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


model = load_model()
metrics = load_metrics()


@app.route("/")
def home():
    # Show last known ROC-AUC if available
    return render_template("index.html", metrics=metrics)


def _ordered_features(form_data):
    """Extract and order incoming form data to match the model's expected feature order."""
    # Normalize form keys to map user input names to model features
    normalized = {k.lower().replace(" ", "").replace("%", ""): v for k, v in form_data.items()}
    mapping = {
        "ffmc": "FFMC",
        "dmc": "DMC",
        "dc": "DC",
        "isi": "ISI",
        "temperature": "temp",
        "temp": "temp",
        "rh": "RH",
        "rh%": "RH",
        "windkm/hr": "wind",
        "windkmhr": "wind",
        "wind": "wind",
        "rain": "rain",
    }

    values = {}
    for key_norm, val in normalized.items():
        if key_norm in mapping:
            values[mapping[key_norm]] = float(val)

    ordered = [values.get(feat, 0.0) for feat in FEATURE_ORDER]
    return np.array([ordered])


@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        features = _ordered_features(request.form)
        proba = model.predict_proba(features)[0][1]
        output = f"{proba:.2f}"

        if proba > 0.5:
            message = f"Your Forest is in Danger.\nProbability of fire occurring is {output}"
        else:
            message = f"Your Forest is safe.\n Probability of fire occurring is {output}"

        return render_template("index.html", pred=message, metrics=metrics)
    except Exception as exc:
        return render_template("index.html", pred=f"Error: {exc}", metrics=metrics), 400


if __name__ == "__main__":
    app.run(debug=True)
