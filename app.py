from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Model Load
model_path = os.path.join(os.getcwd(), "best_rf_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract Features
        features = np.array([
            data["cropType"], data["soilType"], data["nitrogen"],
            data["phosphorus"], data["temperature"], data["humidity"],
            data["pesticides"], data["rainfall"]
        ]).reshape(1, -1)

        # Prediction
        prediction = model.predict(features)[0]

        return jsonify({"yield": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
