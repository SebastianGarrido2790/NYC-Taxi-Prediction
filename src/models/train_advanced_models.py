import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime

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
EXPERIMENT_NAME = "NYC_Taxi_Demand_Advanced"
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    """Load the train and test datasets."""
    logger.info("Loading train and test datasets...")

    train_file = DATA_PROCESSED_DIR / "train_dataset.parquet"
    test_file = DATA_PROCESSED_DIR / "test_dataset.parquet"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "Train or test dataset not found in processed directory."
        )

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    logger.info(f"Loaded training dataset with {len(train_df)} rows.")
    logger.info(f"Loaded testing dataset with {len(test_df)} rows.")

    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare the data for modeling by selecting features and target."""
    logger.info("Preparing data for modeling...")

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
        "lag_3_ride_count",
        "lag_6_ride_count",
        "lag_12_ride_count",
        "lag_24_ride_count",
        "rolling_mean_3h",
        "rolling_mean_24h",
    ]

    X_train = train_df[feature_columns].astype("float64")
    y_train = train_df["target"].astype("float64")
    X_test = test_df[feature_columns].astype("float64")
    y_test = test_df["target"].astype("float64")

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
        f"Prepared training data with {len(X_train)} rows and {len(feature_columns)} features."
    )
    logger.info(
        f"Prepared testing data with {len(X_test)} rows and {len(feature_columns)} features."
    )

    return X_train, y_train, X_test, y_test, feature_columns


def evaluate_model(
    model, X_train, y_train, X_test, y_test, model_name="model", cv_folds=5
):
    """Evaluate the model on both train and test sets with time-series cross-validation on the training set."""
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


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train an XGBoost model with two-step hyperparameter tuning."""
    logger.info("Training XGBoost model with two-step hyperparameter tuning...")

    # Use a small sample of X_train as input example for MLflow
    input_example = X_train.head(5)

    # Step 1: Tune learning rate with a fixed number of estimators
    logger.info("Step 1: Tuning learning rate...")
    learning_rate_grid = {
        "learning_rate": [0.3, 0.4, 0.5, 0.6, 0.7],
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
                max_depth=6,  # Default value
                subsample=1.0,  # Default value
                colsample_bytree=1.0,  # Default value
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
        "n_estimators": [200, 300],  # Expanded range since learning rate is fixed
        "max_depth": [7, 9],
        "subsample": [0.8, 1.0],
        "min_child_weight": [1, 3],  # Added to control overfitting
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
            _, _, _, test_mae, _, _, cv_mae_mean, cv_mae_std = evaluate_model(
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
            mlflow.log_metric("test_mae", test_mae)

            # Update best model if CV MAE is better
            if cv_mae_mean < best_mae:
                best_mae = cv_mae_mean
                best_params = params
                best_model = model

    return best_model, best_params, best_mae, input_example


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train a LightGBM model with hyperparameter tuning."""
    logger.info("Training LightGBM model with hyperparameter tuning...")

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [7, 9],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "min_data_in_leaf": [20, 50],
    }

    best_mae = float("inf")
    best_params = None
    best_model = None

    # Use a small sample of X_train as input example for MLflow
    input_example = X_train.head(5)

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params(params)

            # Train the model
            model = lgb.LGBMRegressor(objective="mae", random_state=42, **params)

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
                model_name=f"LightGBM with params {params}",
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

    return best_model, best_params, best_mae, input_example


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
    params: dict,
    cv_mae_mean: float = None,
    cv_mae_std: float = None,
):
    """Save the model and its metadata locally and log to MLflow."""
    logger.info(f"Saving model and metadata for {model_name}...")

    # Define file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"{model_name}.joblib"
    metadata_file = MODELS_DIR / f"{model_name}_metadata.json"

    # Save the model locally
    joblib.dump(model, model_file)
    logger.info(f"Saved model to {model_file}")

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "features": features,
        "parameters": params,
        "training_timestamp": timestamp,
        "metrics": {
            "train_mae": train_mae,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
        },
        "cross_validation": {
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
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
    mlflow.log_params(params)
    mlflow.log_param("features", features)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_rmse", test_rmse)
    if cv_mae_mean is not None:
        mlflow.log_metric("cv_mae_mean", cv_mae_mean)
    if cv_mae_std is not None:
        mlflow.log_metric("cv_mae_std", cv_mae_std)

    # Log the metadata file as an artifact
    mlflow.log_artifact(metadata_file)


def main():
    logger.info("Starting advanced model training pipeline...")

    # Load data
    train_df, test_df = load_data()

    # Prepare data
    X_train, y_train, X_test, y_test, feature_columns = prepare_data(train_df, test_df)

    # Train XGBoost
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model, xgb_params, xgb_best_mae, xgb_input_example = train_xgboost(
            X_train, y_train, X_test, y_test
        )
        # Re-evaluate the best model to get all metrics
        (
            xgb_train_mae,
            xgb_train_mse,
            xgb_train_rmse,
            xgb_test_mae,
            xgb_test_mse,
            xgb_test_rmse,
            xgb_cv_mae_mean,
            xgb_cv_mae_std,
        ) = evaluate_model(
            xgb_model, X_train, y_train, X_test, y_test, model_name="Best XGBoost"
        )
        save_model_and_metadata(
            xgb_model,
            model_name="xgboost",
            features=feature_columns,
            train_mae=xgb_train_mae,
            train_mse=xgb_train_mse,
            train_rmse=xgb_train_rmse,
            test_mae=xgb_test_mae,
            test_mse=xgb_test_mse,
            test_rmse=xgb_test_rmse,
            X_train=X_train,
            params=xgb_params,
            cv_mae_mean=xgb_cv_mae_mean,
            cv_mae_std=xgb_cv_mae_std,
        )
        mlflow.xgboost.log_model(
            xgb_model, "xgboost_model", input_example=xgb_input_example
        )
        logger.info(
            f"Best XGBoost model: CV MAE = {xgb_best_mae:.4f}, Params = {xgb_params}"
        )

    # Train LightGBM
    with mlflow.start_run(run_name="LightGBM"):
        lgb_model, lgb_params, lgb_best_mae, lgb_input_example = train_lightgbm(
            X_train, y_train, X_test, y_test
        )
        # Re-evaluate the best model to get all metrics
        (
            lgb_train_mae,
            lgb_train_mse,
            lgb_train_rmse,
            lgb_test_mae,
            lgb_test_mse,
            lgb_test_rmse,
            lgb_cv_mae_mean,
            lgb_cv_mae_std,
        ) = evaluate_model(
            lgb_model, X_train, y_train, X_test, y_test, model_name="Best LightGBM"
        )
        save_model_and_metadata(
            lgb_model,
            model_name="lightgbm",
            features=feature_columns,
            train_mae=lgb_train_mae,
            train_mse=lgb_train_mse,
            train_rmse=lgb_train_rmse,
            test_mae=lgb_test_mae,
            test_mse=lgb_test_mse,
            test_rmse=lgb_test_rmse,
            X_train=X_train,
            params=lgb_params,
            cv_mae_mean=lgb_cv_mae_mean,
            cv_mae_std=lgb_cv_mae_std,
        )
        mlflow.lightgbm.log_model(
            lgb_model, "lightgbm_model", input_example=lgb_input_example
        )
        logger.info(
            f"Best LightGBM model: CV MAE = {lgb_best_mae:.4f}, Params = {lgb_params}"
        )

    logger.info("Advanced model training completed and logged to MLflow.")


if __name__ == "__main__":
    main()
