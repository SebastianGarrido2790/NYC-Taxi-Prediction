# Batch-Scoring System Improvement Report: NYC Taxi Demand Prediction

## Overview

The Batch-Scoring System for NYC Taxi Demand Prediction has been successfully implemented, comprising four scripts: `00_data_ingestor.py`, `01_feature_pipeline.py`, `02_training_pipeline.py`, and `03_inference_pipeline.py`. The system ingests raw taxi trip data, engineers features, trains an XGBoost model, and generates hourly demand predictions for Manhattan zones. While the pipeline is functional and produces predictions (e.g., for 2019-04-01 00:00:00), there are several areas where improvements can enhance its accuracy, scalability, and production readiness. This report outlines potential improvements and provides actionable recommendations for future work, including code snippets to illustrate key changes.

---

## Current State of the Batch-Scoring System

**Performance Metrics (from `xgboost_metadata.json`)**

- Training Metrics:
    - train_mae: 16.43
    - train_mse: 1197.37
    - train_rmse: 34.60
- Test Metrics:
    - test_mae: 38.09
    - test_mse: 5221.39
    - test_rmse: 72.26
- Cross-Validation Metrics:
    - cv_mae_mean: 26.62
    - cv_mae_std: 8.03
- Training Data Size: 8,349 rows (6 days of data: 2019-03-24 23:00:00 to 2019-03-30 23:00:00)
- Feature Count: 17

**Key Observations**

- Overfitting: The significant gap between train_mae (16.43) and test_mae (38.09) indicates overfitting, likely due to the small training dataset and the model’s complexity (max_depth=7, n_estimators=200).
- Prediction Accuracy: The test_mae of 38.09 means predictions can be off by 38 rides on average, which is substantial for low-demand zones (e.g., Zone 12 predicted 1 ride) and less impactful but still notable for high-demand zones (e.g., Zone 249 predicted 292 rides at midnight).
- Feature Limitations: The inference pipeline uses hardcoded weather data, and the feature set lacks some potentially useful temporal and spatial features (e.g., weekly lags, nightlife indicators).
- Scalability: While the pipeline handles the current dataset well, it may struggle with larger datasets or real-time requirements in production.

---

## Potential Improvements and Recommendations

### 1. Improve Model Performance

**1.1 Expand Training Data**

- Issue: The current training dataset (8,349 rows, 6 days) is too small to capture long-term patterns (e.g., weekly or seasonal trends), contributing to overfitting.

- Recommendation: Extend the historical backfill in 00_data_ingestor.py to include data from 2018 (e.g., January to December 2018), increasing the training data size to ~12 months. This will help the model learn more robust patterns.

**Code Snippet (Update `00_data_ingestor.py`):**

```python
# In 00_data_ingestor.py
def historical_backfill():
    logger.info("Starting historical backfill...")
    years = [2018, 2019]  # Add 2018 data
    months = list(range(1, 13)) if years == 2018 else list(range(1, 4))
    for year in years:
        for month in months:
            raw_file = RAW_DATA_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"
            if not raw_file.exists():
                logger.warning(f"Raw data file {raw_file} not found. Skipping...")
                continue
            df = dd.read_parquet(raw_file)
            # Filter for the month and clean as before
            start_date = pd.to_datetime(f"{year}-{month:02d}-01")
            end_date = (start_date + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)
            df = df[
                (df["tpep_pickup_datetime"] >= start_date) &
                (df["tpep_pickup_datetime"] < end_date)
            ]
            cleaned_df = clean_data(df)
            output_file = INTERIM_DATA_DIR / f"cleaned_yellow_tripdata_{year}-{month:02d}.parquet"
            cleaned_df.to_parquet(output_file, engine="pyarrow", compression="snappy")
            logger.info(f"Saved cleaned historical data to {output_file}")
```

- Impact: More training data will reduce overfitting by providing a broader range of demand patterns, potentially lowering the test_mae.

**1.2 Enhance Hyperparameter Tuning**

- Issue: The current XGBoost model (learning_rate=0.05, max_depth=7, min_child_weight=5, n_estimators=200, subsample=0.8) overfits, and the two-step tuning process only explores a limited hyperparameter space.
- Recommendation: Expand the hyperparameter grid to include regularization parameters (lambda, alpha) and use a more efficient tuning method (e.g., Bayesian optimization with optuna) to find better parameters.

**Code Snippet (Update `02_training_pipeline.py`):**

```python
import optuna

def objective(trial, X_train, y_train, cv_folds=5):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.7, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "lambda": trial.suggest_float("lambda", 0, 1.0),  # L2 regularization
        "alpha": trial.suggest_float("alpha", 0, 1.0),   # L1 regularization
        "objective": "reg:absoluteerror",
        "random_state": 42,
    }
    model = xgb.XGBRegressor(**params)
    _, _, _, _, _, _, cv_mae_mean, _ = evaluate_model(model, X_train, y_train, X_train, y_train, model_name="XGBoost")
    return cv_mae_mean

def train_xgboost(X_train, X_test, y_train, y_test):
    logger.info("Training XGBoost model with Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    best_params = study.best_params
    best_model = xgb.XGBRegressor(**best_params)
    final_metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test, model_name="Final XGBoost Model")
    train_mae, train_mse, train_rmse, test_mae, test_mse, test_rmse, cv_mae_mean, cv_mae_std = final_metrics
    return best_model, best_params, train_mae, test_mae, X_train.head(5), {
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
    }
```

- Impact: Better hyperparameters will reduce overfitting, potentially lowering the test_mae closer to the cv_mae_mean (26.62).

**1.3 Experiment with Alternative Models**

- Issue: XGBoost, while effective, may not be the best model for time-series forecasting due to its inability to directly model temporal dependencies.
- Recommendation: Experiment with models designed for time-series data, such as LightGBM (faster training) or a neural network like LSTM (captures temporal dependencies).

**Code Snippet (Add LightGBM Option in `02_training_pipeline.py`):**

```python
import lightgbm as lgb

def train_lightgbm(X_train, X_test, y_train, y_test):
    logger.info("Training LightGBM model...")
    params = {
        "objective": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_weight": 5,
        "subsample": 0.8,
        "random_state": 42,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=200)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    return model, params, train_mae, test_mae, X_train.head(5), {
        "train_mae": train_mae,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_mae": test_mae,
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "cv_mae_mean": train_mae,  # Simplified for this example
        "cv_mae_std": 0.0,
        "training_data_size": len(X_train),
        "feature_count": len(X_train.columns),
    }
```

- Impact: LightGBM may train faster and generalize better, while an LSTM could capture temporal patterns more effectively, potentially reducing the test_mae.

---

### 2. Enhance Feature Engineering

**2.1 Integrate Real-Time Weather Data**

- Issue: The inference pipeline (03_inference_pipeline.py) uses hardcoded weather data, which is unrealistic for production and may lead to inaccurate predictions.
- Recommendation: Integrate real-time weather data using an API (e.g., OpenWeatherMap) for historical weather data for 2019-04-01 00:00:00. For production, fetch live weather data.

**Code Snippet (Update `03_inference_pipeline.py`):**

```python
import requests

def fetch_weather_data(timestamp):
    logger.info(f"Fetching weather data for {timestamp}...")
    # Example: Use OpenWeatherMap historical API (requires API key)
    api_key = "your_api_key"
    lat, lon = 40.7128, -74.0060  # NYC coordinates
    dt = int(timestamp.timestamp())
    url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={dt}&appid={api_key}&units=imperial"
    response = requests.get(url)
    data = response.json()
    weather = {
        "temperature": data["current"]["temp"],
        "humidity": data["current"]["humidity"],
        "wind_speed": data["current"]["wind_speed"],
        "cloud_cover_numeric": data["current"]["clouds"],
        "amount_of_precipitation": data["current"].get("rain", {}).get("1h", 0.0),
    }
    return weather

def prepare_next_hour_features(df, manhattan_zones, target_timestamp):
    # ... existing code ...
    # Replace hardcoded weather with API call
    weather = fetch_weather_data(target_timestamp)
    next_hour_df["temperature"] = weather["temperature"]
    next_hour_df["humidity"] = weather["humidity"]
    next_hour_df["wind_speed"] = weather["wind_speed"]
    next_hour_df["cloud_cover_numeric"] = weather["cloud_cover_numeric"]
    next_hour_df["amount_of_precipitation"] = weather["amount_of_precipitation"]
    # ... rest of the code ...
```

- Impact: Accurate weather data will improve prediction quality, as weather significantly impacts taxi demand (e.g., rain increases demand).

**2.2 Add More Temporal and Spatial Features**

- Issue: The current feature set lacks features to capture weekly patterns (e.g., lag of 168 hours for the same hour last week) and spatial characteristics (e.g., nightlife zones).
- Recommendation: Add a weekly lag (lag_168_ride_count) and a is_nightlife feature to indicate zones with high nightlife activity (e.g., West Village, Lower East Side).

**Code Snippet (Update 01_feature_pipeline.py and `03_inference_pipeline.py`):**

```python
# In 01_feature_pipeline.py
def create_features(df):
    # ... existing code ...
    # Add weekly lag
    df["lag_168_ride_count"] = df.groupby("PULocationID")["ride_count"].shift(168)  # 7 days
    df["lag_168_ride_count"] = df["lag_168_ride_count"].fillna(0).astype("float64")
    # Add nightlife indicator
    nightlife_zones = [4, 249, 148]  # Alphabet City, West Village, Lower East Side
    df["is_nightlife"] = df["PULocationID"].isin(nightlife_zones).astype("int8")
    return df

# In 03_inference_pipeline.py
def prepare_next_hour_features(df, manhattan_zones, target_timestamp):
    # ... existing code ...
    # Add weekly lag
    lag_168_timestamp = target_timestamp - pd.Timedelta(hours=168)
    lag_168_data = df[df["hourly_timestamp"] == lag_168_timestamp].set_index("PULocationID")
    next_hour_df["lag_168_ride_count"] = next_hour_df["PULocationID"].map(lag_168_data["ride_count"]).fillna(0).astype("float64")
    # Add nightlife indicator
    nightlife_zones = [4, 249, 148]
    next_hour_df["is_nightlife"] = next_hour_df["PULocationID"].isin(nightlife_zones).astype("int8")
    return next_hour_df
```

- Impact: These features will help the model capture weekly patterns and better predict demand in nightlife zones, especially at midnight.

---

### 3. Evaluate and Monitor Predictions

**3.1 Compare Predictions to Actuals**

- Issue: The predictions for 2019-04-01 00:00:00 have not been compared to actual ride counts, so we don’t know the true inference error.
- Recommendation: Load the actual ride counts from the raw data (yellow_tripdata_2019-04.parquet) and compute the inference MAE.

**Code Snippet (Analysis Script):**

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load predictions
PREDICTIONS_DIR = Path("models/batch-scoring_system/predictions")
predictions = pd.read_parquet(PREDICTIONS_DIR / "predictions_20190401_000000.parquet")

# Load actual data
raw_data = pd.read_parquet("data/raw/yellow_tripdata_2019-04.parquet")
actuals = raw_data[
    (raw_data["tpep_pickup_datetime"] >= "2019-04-01 00:00:00") &
    (raw_data["tpep_pickup_datetime"] < "2019-04-01 01:00:00")
]
actual_counts = actuals.groupby("PULocationID").size().reset_index(name="actual_ride_count")

# Merge and compute MAE
merged = predictions.merge(actual_counts, on="PULocationID", how="left")
merged["actual_ride_count"] = merged["actual_ride_count"].fillna(0)
mae = mean_absolute_error(merged["actual_ride_count"], merged["predicted_ride_count"])
print(f"Inference MAE for 2019-04-01 00:00:00: {mae:.2f}")
```

- Impact: This will provide a true measure of the model’s performance in the inference scenario, allowing us to assess whether the test_mae (38.09) is representative of real-world performance.

**3.2 Implement Monitoring and Logging**

- Issue: The pipeline lacks monitoring to track prediction quality over time in production.
- Recommendation: Add logging of prediction errors and model drift metrics (e.g., feature distribution changes) using a monitoring tool like Prometheus or a custom logging solution.

**Code Snippet (Update `03_inference_pipeline.py`):**

```python
import logging
from sklearn.metrics import mean_absolute_error

def save_predictions(df, actuals=None):
    output_file = PREDICTIONS_DIR / f"predictions_{df['hourly_timestamp'].iloc[0].strftime('%Y%m%d_%H%M%S')}.parquet"
    logger.info(f"Saving predictions to {output_file}...")
    df[["hourly_timestamp", "PULocationID", "predicted_ride_count"]].to_parquet(
        output_file, engine="pyarrow", compression="snappy", index=False
    )
    if actuals is not None:
        merged = df.merge(actuals, on="PULocationID", how="left")
        merged["actual_ride_count"] = merged["actual_ride_count"].fillna(0)
        mae = mean_absolute_error(merged["actual_ride_count"], merged["predicted_ride_count"])
        logger.info(f"Prediction MAE: {mae:.2f}")
        # Log to a monitoring system (e.g., Prometheus)
        with open("prediction_metrics.log", "a") as f:
            f.write(f"{df['hourly_timestamp'].iloc[0]} - MAE: {mae:.2f}\n")
    logger.info(f"Saved predictions to {output_file}")
```

- Impact: Monitoring will enable proactive identification of performance degradation, ensuring the system remains reliable in production.

---

### 4. Production Deployment and Scalability

**4.1 Dynamic Timestamp for Inference**

- Issue: The inference pipeline hardcodes the target timestamp (2019-04-01 00:00:00), which is not suitable for production where predictions should be made for the next hour dynamically.
- Recommendation: Modify 03_inference_pipeline.py to predict for the next hour based on the latest timestamp in the feature store.

**Code Snippet (Update `03_inference_pipeline.py`):**

```python
def main():
    logger.info("Starting inference pipeline...")
    MANHATTAN_ZONES = [...]  # Existing list
    df = load_latest_features()
    latest_timestamp = df["hourly_timestamp"].max()
    target_timestamp = latest_timestamp + pd.Timedelta(hours=1)
    logger.info(f"Predicting for {target_timestamp}...")
    next_hour_df = prepare_next_hour_features(df, MANHATTAN_ZONES, target_timestamp)
    model, metadata = load_model_from_registry()
    predictions_df = make_predictions(next_hour_df, model, metadata)
    save_predictions(predictions_df)
    logger.info("Inference pipeline completed.")
```

- Impact: This makes the pipeline suitable for real-time forecasting, predicting the next hour dynamically.

**4.2 Deploy in a Cloud Environment**

- Issue: The pipeline runs locally, which limits scalability and reliability for production use.
- Recommendation: Deploy the pipeline on a cloud platform (e.g., AWS) using a scheduler (e.g., AWS Lambda with EventBridge) to run the inference pipeline hourly. Use S3 for data storage and SageMaker for model hosting.

**Deployment Plan:**

1. Store Data in S3:
    - Upload raw data, feature store, and predictions to S3 buckets.
2. Host Model in SageMaker:
    - Deploy the trained model as a SageMaker endpoint for inference.
3. Schedule Pipeline with AWS Lambda:
    - Create a Lambda function to run 03_inference_pipeline.py hourly, triggered by EventBridge.
4. Monitor with CloudWatch:
    - Log metrics (e.g., prediction MAE, execution time) to CloudWatch for monitoring.

**Code Snippet (AWS Lambda Handler for Inference):**

```python
import boto3
import pandas as pd
from io import BytesIO

def lambda_handler(event, context):
    s3 = boto3.client("s3")
    # Load feature store from S3
    bucket = "nyc-taxi-data"
    feature_file = "feature_store/features.parquet"
    obj = s3.get_object(Bucket=bucket, Key=feature_file)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    # Run inference pipeline
    latest_timestamp = df["hourly_timestamp"].max()
    target_timestamp = latest_timestamp + pd.Timedelta(hours=1)
    next_hour_df = prepare_next_hour_features(df, MANHATTAN_ZONES, target_timestamp)
    # Call SageMaker endpoint for predictions
    sagemaker = boto3.client("sagemaker-runtime")
    response = sagemaker.invoke_endpoint(
        EndpointName="nyc-taxi-demand-xgboost",
        Body=next_hour_df[metadata["feature_columns"]].to_csv(index=False),
        ContentType="text/csv",
    )
    predictions = pd.read_csv(BytesIO(response["Body"].read()), header=None)
    next_hour_df["predicted_ride_count"] = predictions.round().astype(int)
    # Save predictions to S3
    output_file = f"predictions/predictions_{target_timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
    buffer = BytesIO()
    next_hour_df[["hourly_timestamp", "PULocationID", "predicted_ride_count"]].to_parquet(buffer)
    s3.put_object(Bucket=bucket, Key=output_file, Body=buffer.getvalue())
    return {"statusCode": 200, "body": "Inference completed"}
```

- Impact: Cloud deployment ensures scalability, reliability, and automated hourly predictions, making the system production-ready.

---

## Conclusion

The Batch-Scoring System for NYC Taxi Demand Prediction can be significantly improved by addressing model performance, feature engineering, evaluation, and deployment. Key recommendations include expanding the training data, enhancing hyperparameter tuning, integrating real-time weather data, adding more features, evaluating predictions against actuals, and deploying the pipeline in a cloud environment. These improvements will reduce the `test_mae` (currently 38.09), improve prediction accuracy, and ensure the system is scalable and reliable for production use. Implementing these changes will make the system more effective for operational decision-making in taxi demand forecasting.