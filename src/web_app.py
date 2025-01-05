import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates')

# Paths to model and encoders
model_path = "models/trained_model.pkl"
encoders_path = "models/encoders.pkl"
feature_columns_path = "models/feature_columns.pkl"
API_URL = "http://localhost:5000/predict"  # Change this to your actual prediction API URL

# Function to load the model, encoders, and feature columns
def load_model_and_encoders():
    try:
        logger.debug(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
        if not os.path.exists(encoders_path):
            logger.error(f"Encoders file not found at {encoders_path}")
        if not os.path.exists(feature_columns_path):
            logger.error(f"Feature columns file not found at {feature_columns_path}")
        
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        feature_columns = joblib.load(feature_columns_path)
        
        logger.debug("Model, encoders, and feature columns loaded successfully.")
        return model, encoders, feature_columns
    except Exception as e:
        logger.error(f"Error loading model or encoders: {e}")
        return None, None, None

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        logger.error("No file part in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected.")
        return jsonify({"error": "No selected file"}), 400

    logger.debug(f"File received: {file.filename}")

    # Load model and encoders
    model, encoders, feature_columns = load_model_and_encoders()

    if model is None or encoders is None or feature_columns is None:
        logger.error("Error loading model or encoders.")
        return jsonify({"error": "Error loading model or encoders."}), 500

    # Check file extension (CSV or Excel)
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xls') or file.filename.endswith('.xlsx')):
        # Read the file into a pandas DataFrame
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.xls') or file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            logger.debug(f"File loaded successfully with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return jsonify({"error": f"Error reading file: {str(e)}"}), 500

        # Ensure the file contains the expected columns for prediction
        required_columns = ['sp', 'k', 'e', 'ta', 'pct', 'ast', 'sa', 'se', 're', 'dig', 'bs', 'ba', 'be', 'tb', 'bhe', 'pts', 'year']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {', '.join(missing_cols)}")
            return jsonify({"error": f"Missing columns: {', '.join(missing_cols)}"}), 400

        # Save the 'opponent' column as it is
        opponent_column = df['opponent'].copy()

        # Preprocess data: apply encoders and align columns
        for column, encoder in encoders.items():
            if column in df.columns:
                df[column] = df[column].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
            else:
                df[column] = -1  # Fill missing columns with -1

        # Ensure column order is the same as the model's expected features
        df = df[feature_columns]

        # Predict using the loaded model
        predictions = model.predict(df)
        logger.debug(f"Predictions: {predictions}")

        # Add predictions as the 'win/loss' column
        df['win/loss'] = predictions

        # Add the 'opponent' column back to the dataframe
        df['opponent'] = opponent_column

        # Return the result to the user (pass predictions and the dataset to the template)
        return render_template('index.html', prediction=predictions.tolist(), data=df.to_dict(orient='records'))

    else:
        logger.error("Invalid file format. Please upload a CSV or Excel file.")
        return jsonify({"error": "Invalid file format. Please upload a CSV or Excel file."}), 400

if __name__ == '__main__':
    app.run(debug=True)
