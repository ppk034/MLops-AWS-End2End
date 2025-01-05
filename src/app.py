from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Set paths for the model, encoders, and feature columns
model_path = "/app/models/trained_model.pkl"
encoders_path = "/app/models/encoders.pkl"
feature_columns_path = "/app/models/feature_columns.pkl"

# Verify that all necessary files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(encoders_path):
    raise FileNotFoundError(f"Encoders file not found at {encoders_path}")
if not os.path.exists(feature_columns_path):
    raise FileNotFoundError(f"Feature columns file not found at {feature_columns_path}")

# Load the trained model, encoders, and feature columns
model = joblib.load(model_path)
encoders = joblib.load(encoders_path)
feature_columns = joblib.load(feature_columns_path)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the ML Project API!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Preprocess the data
        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = input_df[column].map(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )

        # Ensure all necessary columns exist
        missing_cols = [col for col in feature_columns if col not in input_df.columns]
        for col in missing_cols:
            input_df[col] = -1  # Add missing columns with default value

        # Drop extra columns not used by the model
        input_df = input_df[feature_columns]

        # Validate input data
        if input_df.shape[1] != len(model.feature_importances_):
            return jsonify({"error": "Mismatch between input features and model expectations."}), 400

        # Make predictions
        prediction = model.predict(input_df)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
