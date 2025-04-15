import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "src" / "models" / "models_and_metadata"
ANALYSIS_DIR = BASE_DIR / "src" / "models" / "analysis"

# Ensure analysis directory exists
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Set up MLflow experiment (same as training)
EXPERIMENT_NAME = "NYC_Taxi_Demand_Advanced"
mlflow.set_experiment(EXPERIMENT_NAME)


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


def load_test_data():
    """Load the test dataset for error analysis."""
    logger.info("Loading test dataset...")

    test_file = DATA_PROCESSED_DIR / "test_dataset.parquet"
    if not test_file.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_file}")

    test_df = pd.read_parquet(test_file)
    logger.info(f"Loaded test dataset with {len(test_df)} rows.")

    return test_df


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


def plot_feature_importance(model, feature_columns, model_name="xgboost"):
    """Plot and log feature importance for the model."""
    logger.info(f"Computing feature importance for {model_name}...")

    # Get feature importance (using 'gain' for XGBoost)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Log feature importance to MLflow
    for _, row in importance_df.iterrows():
        mlflow.log_metric(f"feature_importance_{row['Feature']}", row["Importance"])

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title(f"Feature Importance for {model_name} (Gain)")
    plt.xlabel("Importance (Gain)")
    plt.ylabel("Feature")

    # Save the plot
    plot_path = ANALYSIS_DIR / f"{model_name}_feature_importance.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance plot to {plot_path}")

    # Log the plot to MLflow
    mlflow.log_artifact(plot_path)

    # Save the feature importance DataFrame to CSV
    importance_csv_path = ANALYSIS_DIR / f"{model_name}_feature_importance.csv"
    importance_df.to_csv(importance_csv_path, index=False)
    logger.info(f"Saved feature importance data to {importance_csv_path}")
    mlflow.log_artifact(importance_csv_path)

    return importance_df


def perform_error_analysis(test_df, y_test, y_pred, model_name="xgboost"):
    """Perform error analysis on the test set predictions."""
    logger.info(f"Performing error analysis for {model_name}...")

    # Create a DataFrame with actuals, predictions, and errors
    error_df = test_df[
        [
            "hourly_timestamp",
            "PULocationID",
            "zone",
            "is_rush_hour",
            "is_holiday",
            "is_weekend",
        ]
    ].copy()
    error_df["actual"] = y_test
    error_df["predicted"] = y_pred
    error_df["absolute_error"] = np.abs(error_df["actual"] - error_df["predicted"])
    error_df["hour"] = error_df["hourly_timestamp"].dt.hour
    error_df["day_of_week"] = error_df["hourly_timestamp"].dt.dayofweek

    # 1. Overall error statistics
    overall_mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Overall MAE: {overall_mae:.4f}")
    mlflow.log_metric("overall_mae", overall_mae)

    # 2. Error by zone (top 10 zones with highest average error)
    error_by_zone = (
        error_df.groupby("zone")["absolute_error"].mean().sort_values(ascending=False)
    )
    top_10_zones = error_by_zone.head(10)
    logger.info("Top 10 zones with highest average absolute error:")
    logger.info(top_10_zones)

    # Plot error by zone
    plt.figure(figsize=(12, 6))
    top_10_zones.plot(kind="bar")
    plt.title("Average Absolute Error by Zone (Top 10)")
    plt.xlabel("Zone")
    plt.ylabel("Average Absolute Error")
    plt.xticks(rotation=45, ha="right")
    plot_path = ANALYSIS_DIR / f"{model_name}_error_by_zone.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error by zone plot to {plot_path}")
    mlflow.log_artifact(plot_path)

    # 3. Error by hour of day
    error_by_hour = error_df.groupby("hour")["absolute_error"].mean()
    plt.figure(figsize=(10, 6))
    error_by_hour.plot()
    plt.title("Average Absolute Error by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Absolute Error")
    plt.grid(True)
    plot_path = ANALYSIS_DIR / f"{model_name}_error_by_hour.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error by hour plot to {plot_path}")
    mlflow.log_artifact(plot_path)

    # 4. Error by day of week
    error_by_day = error_df.groupby("day_of_week")["absolute_error"].mean()
    plt.figure(figsize=(10, 6))
    error_by_day.plot()
    plt.title("Average Absolute Error by Day of Week")
    plt.xlabel("Day of Week (0=Mon, 6=Sun)")
    plt.ylabel("Average Absolute Error")
    plt.grid(True)
    plot_path = ANALYSIS_DIR / f"{model_name}_error_by_day.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error by day plot to {plot_path}")
    mlflow.log_artifact(plot_path)

    # 5. Error during rush hour vs. non-rush hour
    error_by_rush_hour = error_df.groupby("is_rush_hour")["absolute_error"].mean()
    logger.info("Average absolute error by rush hour status:")
    logger.info(error_by_rush_hour)
    plt.figure(figsize=(6, 6))
    error_by_rush_hour.plot(kind="bar")
    plt.title("Average Absolute Error by Rush Hour Status")
    plt.xlabel("Is Rush Hour")
    plt.ylabel("Average Absolute Error")
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"], rotation=0)
    plot_path = ANALYSIS_DIR / f"{model_name}_error_by_rush_hour.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error by rush hour plot to {plot_path}")
    mlflow.log_artifact(plot_path)

    # 6. Error on holidays vs. non-holidays
    error_by_holiday = error_df.groupby("is_holiday")["absolute_error"].mean()
    logger.info("Average absolute error by holiday status:")
    logger.info(error_by_holiday)
    plt.figure(figsize=(6, 6))
    error_by_holiday.plot(kind="bar")
    plt.title("Average Absolute Error by Holiday Status")
    plt.xlabel("Is Holiday")
    plt.ylabel("Average Absolute Error")
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"], rotation=0)
    plot_path = ANALYSIS_DIR / f"{model_name}_error_by_holiday.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error by holiday plot to {plot_path}")
    mlflow.log_artifact(plot_path)

    # 7. Error on weekends vs. weekdays
    error_by_weekend = error_df.groupby("is_weekend")["absolute_error"].mean()
    logger.info("Average absolute error by weekend status:")
    logger.info(error_by_weekend)
    plt.figure(figsize=(6, 6))
    error_by_weekend.plot(kind="bar")
    plt.title("Average Absolute Error by Weekend Status")
    plt.xlabel("Is Weekend")
    plt.ylabel("Average Absolute Error")
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"], rotation=0)
    plot_path = ANALYSIS_DIR / f"{model_name}_error_by_weekend.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error by weekend plot to {plot_path}")
    mlflow.log_artifact(plot_path)

    # Save the error DataFrame for further analysis if needed
    error_csv_path = ANALYSIS_DIR / f"{model_name}_error_analysis.csv"
    error_df.to_csv(error_csv_path, index=False)
    logger.info(f"Saved error analysis data to {error_csv_path}")
    mlflow.log_artifact(error_csv_path)

    return error_df


def main():
    logger.info("Starting model analysis pipeline...")

    with mlflow.start_run(run_name="Model_Analysis"):
        # Load the best model (XGBoost) and its metadata
        model, metadata = load_model_and_metadata(model_name="xgboost")

        # Extract the feature columns from metadata
        feature_columns = metadata["features"]

        # Perform feature importance analysis
        importance_df = plot_feature_importance(
            model, feature_columns, model_name="xgboost"
        )

        # Load the test dataset
        test_df = load_test_data()

        # Prepare the test data
        X_test = prepare_data(test_df, feature_columns)
        y_test = test_df["target"].astype("float64")

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Perform error analysis
        error_df = perform_error_analysis(test_df, y_test, y_pred, model_name="xgboost")

    logger.info("Model analysis pipeline completed.")


if __name__ == "__main__":
    main()
