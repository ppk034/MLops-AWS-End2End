import os
import pandas as pd
import logging
import joblib

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/inference.log"),  # Updated path for logs
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def load_model_and_encoders(model_path, encoders_path, feature_columns_path):
    """
    Load the trained model, encoders, and feature columns from disk.
    """
    try:
        logger.info("Loading model, encoders, and feature columns...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders file not found at {encoders_path}")
        if not os.path.exists(feature_columns_path):
            raise FileNotFoundError(f"Feature columns file not found at {feature_columns_path}")
        
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        feature_columns = joblib.load(feature_columns_path)
        logger.info("Model, encoders, and feature columns loaded successfully.")
        return model, encoders, feature_columns
    except Exception as e:
        logger.error(f"Error loading model, encoders, or feature columns: {e}")
        raise

def preprocess_data(data, encoders, feature_columns):
    """
    Preprocess new data for prediction by encoding categorical features
    and aligning columns with the trained model.
    """
    try:
        logger.info("Starting data preprocessing...")
        for column, encoder in encoders.items():
            if column in data.columns:
                logger.info(f"Encoding column: {column}")
                # Handle unseen labels by assigning -1
                data[column] = data[column].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
            else:
                logger.warning(f"Column '{column}' not found in data. Skipping encoding.")
        
        # Ensure all columns align with the model's features
        missing_cols = [col for col in feature_columns if col not in data.columns]
        for col in missing_cols:
            logger.warning(f"Missing column '{col}' in new data. Filling with default value (-1).")
            data[col] = -1
        
        # Drop any extra columns not in the model
        extra_cols = [col for col in data.columns if col not in feature_columns]
        if extra_cols:
            logger.info(f"Dropping extra columns: {extra_cols}")
            data = data.drop(columns=extra_cols)

        # Ensure column order matches feature_columns
        data = data[feature_columns]

        logger.info("Data preprocessing completed successfully.")
        return data
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def validate_and_predict(model, data):
    """
    Validate data shape and perform predictions.
    """
    try:
        logger.info("Validating data for prediction...")
        predictions = model.predict(data)
        logger.info(f"Predictions completed successfully. Sample predictions: {predictions[:5]}")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    try:
        print("Inference script is running...")  # Debug print
        logger.info("Inference Script Started")
        
        # Paths
        model_path = "/app/models/trained_model.pkl"
        encoders_path = "/app/models/encoders.pkl"
        feature_columns_path = "/app/models/feature_columns.pkl"
        new_data_path = "/app/data/new_data.csv"
        predictions_path = "/app/data/predictions.csv"

        # Load the model, encoders, and feature columns
        model, encoders, feature_columns = load_model_and_encoders(model_path, encoders_path, feature_columns_path)

        # Load new data
        if not os.path.exists(new_data_path):
            logger.error(f"New data file not found at {new_data_path}")
            raise FileNotFoundError(f"File not found: {new_data_path}")
        new_data = pd.read_csv(new_data_path)
        logger.info(f"New data loaded successfully. Shape: {new_data.shape}")
        logger.info(f"Preview of new data:\n{new_data.head()}")  # Log data preview

        # Preprocess data
        preprocessed_data = preprocess_data(new_data.copy(), encoders, feature_columns)

        # Validate feature alignment with the model
        missing_features = set(feature_columns) - set(preprocessed_data.columns)
        extra_features = set(preprocessed_data.columns) - set(feature_columns)
        if missing_features or extra_features:
            logger.error("Feature alignment issues detected!")
            logger.error(f"Missing features: {missing_features}")
            logger.error(f"Extra features: {extra_features}")
            raise ValueError("Mismatch between model features and input data columns.")
        else:
            logger.info("Feature alignment check passed.")

        # Predict
        predictions = validate_and_predict(model, preprocessed_data)

        # Save predictions
        new_data["Predictions"] = predictions
        new_data.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
