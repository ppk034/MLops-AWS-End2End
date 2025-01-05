import pandas as pd
import logging
from logger_config import logger

def preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    
    :param file_path: Path to the dataset file
    :return: Preprocessed DataFrame
    """
    try:
        logger.info(f"Starting preprocessing for file: {file_path}")

        # Load the dataset with correct delimiter
        df = pd.read_csv(file_path, sep='\t')  # Specify tab as the delimiter
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Drop rows with missing values
        df.dropna(inplace=True)
        logger.info(f"Missing values removed. Shape: {df.shape}")

        # Rename columns for consistency
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        logger.info(f"Column names standardized: {list(df.columns)}")

        # Convert W/L column to numeric (1 for W, 0 for L)
        if "w/l" in df.columns:
            logger.info(f"Converting 'w/l' column to numeric values...")
            df["w/l"] = df["w/l"].str.strip().str.upper().map({"W": 1, "L": 0})
            if df["w/l"].isnull().any():
                logger.warning(f"'w/l' column contains unmapped values: {df['w/l'].unique()}")
            else:
                logger.info(f"'w/l' column successfully converted to numeric values.")
        else:
            logger.error(f"'w/l' column not found. Columns available: {list(df.columns)}")

        # Save the cleaned data
        cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")
        df.to_csv(cleaned_file_path, index=False)
        logger.info(f"Cleaned data saved to {cleaned_file_path}")

        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None

if __name__ == "__main__":
    # Define the file path
    file_path = "../data/project_data.csv"
    preprocess_data(file_path)
