import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "src" / "models" / "models_and_metadata"
PREDICTIONS_DIR = BASE_DIR / "src" / "models" / "predictions"

# Ensure predictions directory exists
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_metadata(model_name="xgboost"):
    """Load the trained model and its metadata."""
    logger.info(f"Loading {model_name} model and metadata...")

    model_file = MODELS_DIR / f"{model_name}.joblib"
    metadata_file = MODELS_DIR / f"{model_name}_metadata.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    # Load the model
    model = joblib.load(model_file)
    logger.info(f"Loaded model from {model_file}")

    # Load the metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    logger.info(f"Loaded metadata from {metadata_file}")

    return model, metadata


def load_data(data_path=None):
    """Load the dataset for prediction. If no path is provided, use the test dataset."""
    logger.info("Loading dataset for prediction...")

    if data_path is None:
        data_file = DATA_PROCESSED_DIR / "test_dataset.parquet"
    else:
        data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    df = pd.read_parquet(data_file)
    logger.info(f"Loaded dataset with {len(df)} rows from {data_file}")

    return df


def prepare_data(df, feature_columns):
    """Prepare the data for prediction by selecting the required features."""
    logger.info("Preparing data for prediction...")

    # Select the features used during training
    X = df[feature_columns].astype("float64")

    # Check for NaNs
    if X.isna().sum().sum() > 0:
        logger.error("NaNs found in input data:")
        logger.error(X.isna().sum())
        raise ValueError("Input data contains NaN values.")

    logger.info(
        f"Prepared data with {len(X)} rows and {len(feature_columns)} features."
    )

    return X


def make_predictions(model, X, df):
    """Make predictions using the loaded model."""
    logger.info("Making predictions...")

    # Make predictions
    predictions = model.predict(X)

    # Create a DataFrame with predictions and relevant columns
    result_df = df[["hourly_timestamp", "PULocationID", "zone"]].copy()
    result_df["predicted_ride_count"] = predictions

    logger.info(f"Generated predictions for {len(predictions)} rows.")

    return result_df


def save_predictions(result_df, output_path=None):
    """Save the predictions to a CSV file."""
    logger.info("Saving predictions...")

    if output_path is None:
        output_file = PREDICTIONS_DIR / "predictions.csv"
    else:
        output_file = Path(output_path)

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    result_df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions to {output_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the dataset for prediction. If not provided, uses the test dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the predictions. If not provided, saves to data/predictions/predictions.csv.",
    )
    args = parser.parse_args()

    logger.info("Starting prediction pipeline...")

    # Load the best model (XGBoost) and its metadata
    model, metadata = load_model_and_metadata(model_name="xgboost")

    # Extract the feature columns from metadata
    feature_columns = metadata["features"]

    # Load the dataset
    df = load_data(data_path=args.data_path)

    # Prepare the data
    X = prepare_data(df, feature_columns)

    # Make predictions
    result_df = make_predictions(model, X, df)

    # Save the predictions
    save_predictions(result_df, output_path=args.output_path)

    logger.info("Prediction pipeline completed.")


if __name__ == "__main__":
    main()

# Command-Line Interface
# The script can be run with optional arguments:
# python src/models/predict_model.py --data-path path/to/new_data.parquet --output-path path/to/output.csv
# If no arguments are provided, it uses the test dataset and saves predictions to the default location.
