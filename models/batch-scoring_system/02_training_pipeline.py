import pandas as pd
from pathlib import Path
import logging
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib
import json
from datetime import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_STORE_DIR = BASE_DIR / "models" / "batch-scoring_system" / "feature_store"
MODELS_DIR = BASE_DIR / "models" / "batch-scoring_system" / "models_and_metadata"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set up MLflow experiment
EXPERIMENT_NAME = "NYC_Taxi_Demand_Training_Pipeline"
mlflow.set_experiment(EXPERIMENT_NAME)


def load_features():
    """Load features from the feature store."""
    logger.info("Loading features from feature store...")

    feature_file = FEATURE_STORE_DIR / "features.parquet"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file {feature_file} not found.")

    df = pd.read_parquet(feature_file)
    logger.info(f"Loaded features with {len(df)} rows.")
    return df


def create_time_series_data(df):
    """Create time-series data with target for training."""
    logger.info("Creating time-series data for training...")

    # Sort by PULocationID and hourly_timestamp
    df = df.sort_values(["PULocationID", "hourly_timestamp"])

    # Create target (next hour's ride count)
    df["target"] = df.groupby("PULocationID")["ride_count"].shift(-1)
    df["target"] = df["target"].astype("float64")

    # Drop rows with missing target (last timestamp per zone will have no target)
    df = df.dropna(subset=["target"])
    logger.info(f"Time-series data created with {len(df)} rows.")
    return df


def prepare_data(df):
    """Prepare the data for training with a time-based split."""
    logger.info("Preparing data for training...")

    # Define feature columns (now including precomputed lagged and rolling features)
    feature_columns = [
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "is_weekend",
        "is_holiday",
        "is_rush_hour",
        "temperature",
        "humidity",
        "wind_speed",
        "cloud_cover_numeric",
        "amount_of_precipitation",
        "is_downtown",
        "lag_1_ride_count",
        "lag_2_ride_count",
        "lag_24_ride_count",
        "rolling_mean_3h",
        "rolling_mean_24h",
    ]

    # Ensure no missing values in features or target
    df = df.dropna(subset=feature_columns + ["target"])

    # Time-based split: use 2019-03-24 23:00:00 to 2019-03-30 23:00:00 for training, 2019-03-31 for testing
    split_date = pd.to_datetime("2019-03-30 23:00:00")
    train_df = df[df["hourly_timestamp"] <= split_date]
    test_df = df[df["hourly_timestamp"] > split_date]

    X_train = train_df[feature_columns].astype("float64")
    y_train = train_df["target"].astype("float64")
    X_test = test_df[feature_columns].astype("float64")
    y_test = test_df["target"].astype("float64")

    logger.info(f"Training data: {len(X_train)} rows, Test data: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test, feature_columns


def evaluate_model(
    model, X_train, y_train, X_test, y_test, model_name="model", cv_folds=5
):
    """
    Evaluate the model on both train and test sets with time-series cross-validation on the training set.

    Args:
        model: The model to evaluate
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of evaluation metrics: (train_mae, train_mse, train_rmse, test_mae, test_mse, test_rmse, cv_mae_mean, cv_mae_std)
    """
    logger.info(
        f"Evaluating {model_name} with {cv_folds}-fold time-series cross-validation..."
    )

    # Perform time-series cross-validation on the training set
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    cv_mae_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        logger.info(f"Processing fold {fold}/{cv_folds}...")
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Predict on the validation fold
        y_val_pred = model.predict(X_val_fold)

        # Compute MAE for this fold
        fold_mae = mean_absolute_error(y_val_fold, y_val_pred)
        cv_mae_scores.append(fold_mae)
        logger.info(f"Fold {fold} MAE: {fold_mae:.4f}")

    # Compute mean and standard deviation of cross-validation scores
    cv_mae_scores = np.array(cv_mae_scores)
    cv_mae_mean = cv_mae_scores.mean()
    cv_mae_std = cv_mae_scores.std()

    # Fit the model on the entire training set
    model.fit(X_train, y_train)

    # Make predictions on train and test sets
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

    # Log all metrics
    logger.info(
        f"{model_name} - Cross-Validation Summary ({cv_folds}-fold) on Training Set:"
    )
    logger.info(f"CV Mean MAE: {cv_mae_mean:.4f} (+/- {cv_mae_std:.4f})")
    logger.info(f"{model_name} - Training Set (Full):")
    logger.info(f"Mean Absolute Error (MAE): {train_mae:.4f}")
    logger.info(f"Mean Squared Error (MSE): {train_mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
    logger.info(f"{model_name} - Test Set:")
    logger.info(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    logger.info(f"Mean Squared Error (MSE): {test_mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")

    return (
        train_mae,
        train_mse,
        train_rmse,
        test_mae,
        test_mse,
        test_rmse,
        cv_mae_mean,
        cv_mae_std,
    )


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train an XGBoost model with two-step hyperparameter tuning."""
    logger.info("Training XGBoost model with two-step hyperparameter tuning...")

    # Step 1: Tune learning rate with a fixed number of estimators
    logger.info("Step 1: Tuning learning rate...")
    learning_rate_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "n_estimators": [200],  # Fixed number of estimators for learning rate tuning
    }

    best_lr_mae = float("inf")
    best_lr = None
    best_lr_model = None

    for params in ParameterGrid(learning_rate_grid):
        with mlflow.start_run(nested=True, run_name="XGBoost_Learning_Rate_Tuning"):
            # Log parameters
            mlflow.log_params(params)

            # Train the model with default values for other parameters
            model = xgb.XGBRegressor(
                objective="reg:absoluteerror",
                random_state=42,
                max_depth=7,  # Default value
                subsample=0.8,  # Default value
                min_child_weight=1,  # Default value
                **params,
            )

            # Evaluate the model with cross-validation
            _, _, _, test_mae, _, _, cv_mae_mean, cv_mae_std = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                model_name=f"XGBoost with learning_rate {params['learning_rate']}",
            )

            # Log metrics to MLflow
            mlflow.log_metric("cv_mae_mean", cv_mae_mean)
            mlflow.log_metric("cv_mae_std", cv_mae_std)
            mlflow.log_metric("test_mae", test_mae)

            # Update best learning rate if CV MAE is better
            if cv_mae_mean < best_lr_mae:
                best_lr_mae = cv_mae_mean
                best_lr = params["learning_rate"]
                best_lr_model = model

    logger.info(f"Best learning rate: {best_lr} with CV MAE: {best_lr_mae:.4f}")

    # Step 2: Tune other parameters with the best learning rate
    logger.info("Step 2: Tuning other parameters with best learning rate...")
    param_grid = {
        "learning_rate": [best_lr],  # Use the best learning rate from Step 1
        "n_estimators": [150, 200],  # Expanded range since learning rate is fixed
        "max_depth": [7, 9],
        "subsample": [0.8, 1.0],
        "min_child_weight": [3, 5],  # Added to control overfitting
        "lambda": [0, 0.5],
        "gamma": [0, 0.5],
    }

    best_mae = float("inf")
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(nested=True, run_name="XGBoost_Parameter_Tuning"):
            # Log parameters
            mlflow.log_params(params)

            # Train the model
            model = xgb.XGBRegressor(
                objective="reg:absoluteerror", random_state=42, **params
            )

            # Evaluate the model with cross-validation
            (
                train_mae,
                train_mse,
                train_rmse,
                test_mae,
                test_mse,
                test_rmse,
                cv_mae_mean,
                cv_mae_std,
            ) = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                model_name=f"XGBoost with params {params}",
            )

            # Log metrics to MLflow
            mlflow.log_metric("cv_mae_mean", cv_mae_mean)
            mlflow.log_metric("cv_mae_std", cv_mae_std)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_rmse", test_rmse)

            # Update best model if CV MAE is better
            if cv_mae_mean < best_mae:
                best_mae = cv_mae_mean
                best_params = params
                best_model = model

    # Use the best model to compute final metrics
    final_metrics = evaluate_model(
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="Final XGBoost Model",
    )
    (
        train_mae,
        train_mse,
        train_rmse,
        test_mae,
        test_mse,
        test_rmse,
        cv_mae_mean,
        cv_mae_std,
    ) = final_metrics

    return (
        best_model,
        best_params,
        train_mae,
        test_mae,
        X_train.head(5),
        {
            "train_mae": train_mae,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "training_data_size": len(X_train),
            "feature_count": len(X_train.columns),
        },
    )


def save_model_and_metadata(
    model, model_name, params, train_mae, test_mae, feature_columns, metrics
):
    """Save the model and metadata to the Model Registry."""
    logger.info(f"Saving {model_name} model and metadata...")

    # Save model locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{model_name}_{timestamp}.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": model.__class__.__name__,
        "parameters": params,
        "feature_columns": feature_columns,
        "target_column": "target",
        "training_timestamp": timestamp,
        "metrics": {
            "train_mae": metrics["train_mae"],
            "train_mse": metrics["train_mse"],
            "train_rmse": metrics["train_rmse"],
            "test_mae": metrics["test_mae"],
            "test_mse": metrics["test_mse"],
            "test_rmse": metrics["test_rmse"],
        },
        "cross_validation": {
            "cv_mae_mean": metrics["cv_mae_mean"],
            "cv_mae_std": metrics["cv_mae_std"],
        },
        "training_data_size": metrics["training_data_size"],
        "feature_count": metrics["feature_count"],
    }
    metadata_path = MODELS_DIR / f"{model_name}_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Log to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", metrics["train_mse"])
    mlflow.log_metric("train_rmse", metrics["train_rmse"])
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mse", metrics["test_mse"])
    mlflow.log_metric("test_rmse", metrics["test_rmse"])
    mlflow.log_metric("cv_mae_mean", metrics["cv_mae_mean"])
    mlflow.log_metric("cv_mae_std", metrics["cv_mae_std"])
    mlflow.log_artifact(metadata_path)
    return model_path, metadata_path


def main():
    logger.info("Starting training pipeline...")

    with mlflow.start_run(run_name="XGBoost_Training"):
        # Load features
        df = load_features()

        # Create time-series data
        df = create_time_series_data(df)

        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = prepare_data(df)

        # Train XGBoost with two-step tuning
        model, params, train_mae, test_mae, input_example, metrics = train_xgboost(
            X_train, X_test, y_train, y_test
        )

        # Save model and metadata
        model_path, metadata_path = save_model_and_metadata(
            model, "xgboost", params, train_mae, test_mae, feature_columns, metrics
        )

        # Log model to MLflow
        mlflow.xgboost.log_model(model, "xgboost_model", input_example=input_example)

        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/xgboost_model"
        registered_model_name = "NYC_Taxi_Demand_XGBoost"
        mlflow.register_model(model_uri, registered_model_name)
        logger.info(f"Registered XGBoost model as {registered_model_name}")

    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    main()
