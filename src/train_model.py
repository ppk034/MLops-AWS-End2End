import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Configure logging
logging.basicConfig(
    filename="/app/logs/train_model.log",  # Updated log path
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

def load_data(data_path):
    """
    Load the dataset.
    """
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the data for training, including encoding categorical columns.
    """
    try:
        logger.info("Starting data preprocessing")

        # Drop unnecessary columns
        if "date" in data.columns:
            logger.info("Dropping 'date' column")
            data = data.drop(columns=["date"])

        # Initialize encoders dictionary
        encoders = {}

        # Encode categorical columns
        for col in data.select_dtypes(include=["object"]).columns:
            if col != "w/l":  # Do not encode the target column
                logger.info(f"Encoding column: {col}")
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                encoders[col] = le

        # Check for missing values
        if data.isnull().sum().sum() > 0:
            logger.warning("Dataset contains missing values. Filling with median values.")
            data = data.fillna(data.median())

        # Separate features and target
        if "w/l" not in data.columns:
            raise ValueError("'w/l' column not found in the dataset")

        X = data.drop(columns=["w/l"])
        y = data["w/l"]

        # Save original column names for later use
        feature_columns_path = "/app/models/feature_columns.pkl"  # Updated path
        joblib.dump(X.columns.tolist(), feature_columns_path)
        logger.info(f"Feature columns saved at {feature_columns_path}")

        logger.info("Data preprocessing completed")
        return X, y, encoders
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def train_and_optimize_model(X, y):
    """
    Train and optimize a Random Forest model using GridSearchCV.
    """
    try:
        logger.info("Starting model training and optimization")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

        # Define the model and hyperparameters
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 4],
        }

        logger.info("Performing grid search for hyperparameter optimization")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose=1)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        logger.info(f"Model Evaluation:\n"
                    f"Accuracy: {accuracy:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    f"ROC-AUC: {roc_auc:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Save the model
        model_path = "/app/models/trained_model.pkl"  # Updated path
        joblib.dump(best_model, model_path)
        logger.info(f"Trained model saved at {model_path}")

        return best_model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Train Model Script Started")
        data_path = "/app/data/project_data_cleaned.csv"  # Updated path

        # Log paths to check if files are available
        logger.info(f"Model path: /app/models/trained_model.pkl")
        logger.info(f"Encoders path: /app/models/encoders.pkl")
        logger.info(f"Feature columns path: /app/models/feature_columns.pkl")

        # Load and preprocess data
        data = load_data(data_path)
        X, y, encoders = preprocess_data(data)

        # Save encoders
        encoders_path = "/app/models/encoders.pkl"  # Updated path
        joblib.dump(encoders, encoders_path)
        logger.info(f"Encoders saved at {encoders_path}")

        # Train and optimize the model
        train_and_optimize_model(X, y)
        logger.info("Train Model Script Completed Successfully")
    except Exception as e:
        logger.error(f"Train Model Script Failed: {e}")
