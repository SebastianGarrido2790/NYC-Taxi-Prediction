import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "src" / "models" / "models_and_metadata"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set up MLflow experiment
EXPERIMENT_NAME = "NYC_Taxi_Demand_Baseline"
mlflow.set_experiment(EXPERIMENT_NAME)


class BaselineModelPreviousHour(BaseEstimator, RegressorMixin):
    """
    A baseline model that predicts the next hour's demand as the demand from the previous hour.

    Note: This model doesn't require any training since it's a simple rule-based approach.
    However, we'll wrap it in a class to make it compatible with scikit-learn’s API and MLflow's model logging.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        No training is required for this model.
        """
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict the next hour's demand as the demand from the previous hour (lag_1_ride_count).
        """
        return X_test["lag_1_ride_count"].values


def load_datasets():
    """Load the train and test datasets."""
    logger.info("Loading train and test datasets...")
    train_file = DATA_PROCESSED_DIR / "train_dataset.parquet"
    test_file = DATA_PROCESSED_DIR / "test_dataset.parquet"

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset {train_file} not found.")
    if not test_file.exists():
        raise FileNotFoundError(f"Testing dataset {test_file} not found.")

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    logger.info(f"Loaded training dataset with {len(train_df)} rows.")
    logger.info(f"Loaded testing dataset with {len(test_df)} rows.")
    return train_df, test_df


def analyze_data_distribution(train_df, test_df):
    """Analyze the distribution of ride_count, target, and lag_1_ride_count in the datasets."""
    logger.info("Analyzing data distribution...")

    # Training set
    logger.info("Training set - ride_count distribution:")
    logger.info(train_df["ride_count"].describe())
    logger.info("Training set - lag_1_ride_count distribution:")
    logger.info(train_df["lag_1_ride_count"].describe())
    logger.info("Training set - target distribution:")
    logger.info(train_df["target"].describe())

    # Test set
    logger.info("Test set - ride_count distribution:")
    logger.info(test_df["ride_count"].describe())
    logger.info("Test set - lag_1_ride_count distribution:")
    logger.info(test_df["lag_1_ride_count"].describe())
    logger.info("Test set - target distribution:")
    logger.info(test_df["target"].describe())


def prepare_data(train_df, test_df):
    """Prepare the data for modeling by selecting features and target."""
    logger.info("Preparing data for modeling...")

    # Define features and target (only lag_1_ride_count is needed for this baseline)
    features = ["lag_1_ride_count"]
    target = "target"

    # Ensure all features are numeric
    X_train = train_df[features].astype("float64")
    y_train = train_df[target].astype("float64")
    X_test = test_df[features].astype("float64")
    y_test = test_df[target].astype("float64")

    # Check for NaNs
    if X_train.isna().sum().sum() > 0:
        logger.error("NaNs found in X_train:")
        logger.error(X_train.isna().sum())
        raise ValueError("X_train contains NaN values.")
    if X_test.isna().sum().sum() > 0:
        logger.error("NaNs found in X_test:")
        logger.error(X_test.isna().sum())
        raise ValueError("X_test contains NaN values.")
    if y_train.isna().sum() > 0:
        logger.error("NaNs found in y_train.")
        raise ValueError("y_train contains NaN values.")
    if y_test.isna().sum() > 0:
        logger.error("NaNs found in y_test.")
        raise ValueError("y_test contains NaN values.")

    logger.info(
        f"Prepared training data with {len(X_train)} rows and {len(features)} features."
    )
    logger.info(
        f"Prepared testing data with {len(X_test)} rows and {len(features)} features."
    )
    return X_train, y_train, X_test, y_test, features


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="model"):
    """Evaluate the model on both train and test sets."""
    logger.info(f"Evaluating {model_name} on train and test sets...")

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics for training set
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    # Compute metrics for test set
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    logger.info(f"{model_name} - Training Set:")
    logger.info(f"Mean Absolute Error (MAE): {train_mae:.4f}")
    logger.info(f"Mean Squared Error (MSE): {train_mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
    logger.info(f"{model_name} - Test Set:")
    logger.info(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    logger.info(f"Mean Squared Error (MSE): {test_mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    return train_mae, train_mse, train_rmse, test_mae, test_mse, test_rmse


def save_model_and_metadata(
    model,
    model_name: str,
    features: list,
    train_mae: float,
    train_mse: float,
    train_rmse: float,
    test_mae: float,
    test_mse: float,
    test_rmse: float,
    X_train: pd.DataFrame,
):
    """Save the model and its metadata locally and log to MLflow."""
    logger.info(f"Saving model and metadata for {model_name}...")

    # Define file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"{model_name}_model.joblib"
    metadata_file = MODELS_DIR / f"{model_name}_metadata.json"

    # Save the model locally
    joblib.dump(model, model_file)
    logger.info(f"Saved model to {model_file}")

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "features": features,
        "training_timestamp": timestamp,
        "metrics": {
            "train_mae": train_mae,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
        },
        "training_data_size": len(X_train),
        "feature_count": len(features),
    }

    # Save metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to {metadata_file}")

    # Log the model and metrics to MLflow
    mlflow.log_param("model_type", type(model).__name__)
    mlflow.log_param("features", features)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_rmse", test_rmse)

    # Log the model to MLflow with an input example
    input_example = X_train.head(1)
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

    # Log the metadata file as an artifact
    mlflow.log_artifact(metadata_file)


def main():
    logger.info("Starting baseline model training pipeline...")

    # Start an MLflow run
    with mlflow.start_run(run_name="Baseline_Previous_Hour"):
        # Load datasets
        train_df, test_df = load_datasets()

        # Analyze data distribution
        analyze_data_distribution(train_df, test_df)

        # Prepare data
        X_train, y_train, X_test, y_test, features = prepare_data(train_df, test_df)

        # Initialize and "train" the baseline model (no actual training needed)
        model = BaselineModelPreviousHour()
        model.fit(X_train, y_train)

        # Evaluate the model
        train_mae, train_mse, train_rmse, test_mae, test_mse, test_rmse = (
            evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                model_name="Baseline Previous Hour",
            )
        )

        # Save the model and metadata
        save_model_and_metadata(
            model,
            model_name="baseline_previous_hour",
            features=features,
            train_mae=train_mae,
            train_mse=train_mse,
            train_rmse=train_rmse,
            test_mae=test_mae,
            test_mse=test_mse,
            test_rmse=test_rmse,
            X_train=X_train,
        )

        logger.info("Baseline model evaluation completed and logged to MLflow.")


if __name__ == "__main__":
    main()
