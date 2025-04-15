# Batch-Scoring System Report: NYC Taxi Demand Prediction

## Project Overview

The Batch-Scoring System is a machine learning pipeline designed to predict hourly taxi demand in Manhattan zones for the NYC Yellow Taxi dataset. The system processes historical and near-real-time data, extracts relevant features, trains a predictive model, and generates hourly demand forecasts for operational use. The pipeline is structured into four scripts, each handling a distinct stage of the data processing and modeling workflow:

**1. Data Ingestion (`00_data_ingestor.py`):** Ingests and cleans raw taxi trip data, preparing it for downstream feature engineering.

**2. Feature Engineering (`01_feature_pipeline.py`):** Transforms cleaned data into a feature store with aggregated and engineered features for modeling.

**3. Model Training (`02_training_pipeline.py`):** Trains an XGBoost model to predict hourly taxi demand, using time-series cross-validation and hyperparameter tuning.

**4. Inference (`03_inference_pipeline.py`):** Generates predictions for a specified future timestamp (2019-04-01 00:00:00) using the trained model.

The project aims to provide accurate, scalable, and production-ready demand forecasts for taxi operators, leveraging temporal, spatial, and external features (e.g., weather) to capture demand patterns. The system was developed iteratively, addressing challenges such as data quality, temporal dependencies, and model overfitting.

---

## Script Details

### 1. Data Ingestion (`00_data_ingestor.py`)

**Purpose**

The 00_data_ingestor.py script is the entry point of the pipeline, responsible for ingesting raw NYC Yellow Taxi trip data, cleaning it, and saving it in a processed format for downstream use. It handles both historical backfills (e.g., January to March 2019) and near-real-time data (e.g., the latest hour: 2019-03-31 23:00:00 to 2019-04-01 00:00:00).

**Reasoning and Logic**

- Input Data: The raw data consists of trip records with columns like tpep_pickup_datetime, PULocationID, passenger_count, trip_distance, fare_amount, and others.

- Steps:
    1. Historical Backfill: Loads data for January to March 2019, filtering for trips within each month.

    2. Latest Data: Extracts the latest hour of data (2019-03-31 23:00:00 to 2019-04-01 00:00:00) from the March dataset.

    3. Data Cleaning:
        - Filters for Manhattan zones (69 zones identified using a predefined list).

        - Ensures timestamp consistency by filtering out trips outside the desired time window.

        - Handles missing values: drops airport_fee (100% missing), imputes numerical columns (passenger_count, trip_distance, fare_amount, total_amount) with medians, and fills congestion_surcharge with 0 (as it was introduced in 2019 and often missing).

        - Validates categorical columns (VendorID, RatecodeID, payment_type) against expected values.

        - Removes outliers: drops rows with negative fares, trip distances > 100 miles, or passenger counts > 6.

    4. Output: Saves cleaned data as Parquet files (e.g., cleaned_yellow_tripdata_2019-01.parquet) in the data/interim/production directory.

- Scalability: Uses Dask for parallel processing of large datasets (e.g., January 2019: 7,684,831 rows), repartitioning the data into 7 partitions for efficiency.

**Challenges**

- Data Volume: The raw datasets are large (e.g., January 2019 has 7.68 million rows), requiring efficient processing. Dask was chosen over Pandas to handle memory constraints, but outlier removal was still time-intensive (e.g., ~6 minutes for January).

- Data Quality: Missing values (e.g., airport_fee, congestion_surcharge) and outliers (e.g., negative fares) required careful handling to avoid skewing downstream models.
 
- Temporal Consistency: Ensuring that timestamps were within the desired windows (e.g., monthly for historical data, hourly for the latest data) was critical to avoid data leakage.

- Manhattan Filtering: Identifying and filtering for Manhattan zones required a predefined list of zone IDs, which needed to be consistent across all scripts.

**Output**

- Historical data: January (7,152,331 rows, 92.93% retention), February (6,629,806 rows, 94.05% retention), March (7,412,679 rows, 94.23% retention).

- Latest data: 4,693 rows for 2019-03-31 23:00:00 to 2019-04-01 00:00:00.

- Total cleaned data: 21,199,509 rows across all files.

---

### 2. Feature Engineering (`01_feature_pipeline.py`)

**Purpose**

The 01_feature_pipeline.py script transforms the cleaned trip data into a feature store, aggregating ride counts by hour and zone, and engineering features for modeling. It creates a complete grid of all Manhattan zones across the time period, ensuring no missing data points for training and inference.

**Reasoning and Logic**

- Input Data: The cleaned Parquet files from 00_data_ingestor.py (January to March 2019, plus the latest hour).

- Steps:
    1. Time Window: Processes the last 7 days + 1 hour (2019-03-24 23:00:00 to 2019-03-31 23:00:00, 169 hours).

    2. Aggregation: Groups the data by hourly_timestamp (rounded to the hour) and PULocationID, counting the number of rides (ride_count) per group.

    3. Grid Creation: Creates a complete grid for all 69 Manhattan zones over 169 hours (11,661 rows = 69 zones × 169 hours), filling missing zone-hour combinations with ride_count=0.

    4. Feature Engineering:
        - Temporal Features: Adds hour_of_day, day_of_week, day_of_month, is_weekend, is_holiday, is_rush_hour to capture time-based patterns.

        - Zone Features: Adds is_downtown (binary indicator for downtown zones like Lower Manhattan and Midtown) and zone (numerical zone ID).

        - Weather Features: Merges with external weather data (nyc_weather.csv), adding temperature, humidity, wind_speed, cloud_cover_numeric, and amount_of_precipitation. Missing weather data is forward-filled.

    5. Output: Saves the feature store as features.parquet with 11,661 rows and 16 columns.

**Challenges**

- Missing Data Points: Some zone-hour combinations had no trips, resulting in missing rows after aggregation. The grid creation step ensured all combinations were present, filling missing ride_count with 0 to avoid bias in modeling.

- Weather Data Integration: The weather dataset (nyc_weather.csv) had missing hours (only 56 rows), requiring forward-filling to align with the 169-hour window. In production, a more robust weather data source would be needed.

- Feature Selection: Deciding which features to include required balancing complexity and predictive power. Features like is_rush_hour and is_downtown were added to capture domain-specific patterns (e.g., higher demand during rush hours or in downtown areas).

- Scalability: While the feature engineering step was less computationally intensive than data ingestion, ensuring the grid creation was efficient for large time windows was critical.

**Output**

- Feature store (features.parquet): 11,661 rows (169 hours × 69 zones) with 16 columns: hourly_timestamp, PULocationID, ride_count, hour_of_day, day_of_week, day_of_month, is_weekend, is_holiday, is_rush_hour, temperature, humidity, wind_speed, cloud_cover_numeric, amount_of_precipitation, is_downtown, zone.

---

### 3. Model Training (`02_training_pipeline.py`)

**Purpose**

The 02_training_pipeline.py script trains an XGBoost model to predict the next hour’s taxi demand (ride_count) for each Manhattan zone, using the features from the feature store. It employs time-series cross-validation and a two-step hyperparameter tuning process to ensure robust model performance.

**Reasoning and Logic**

- Input Data: The feature store (features.parquet) with 11,661 rows.

- Steps:
    1. Time-Series Data Preparation:
        - Adds lagged features (lag_1_ride_count, lag_2_ride_count, lag_24_ride_count) to capture historical demand patterns.

        - Adds rolling mean features (rolling_mean_3h, rolling_mean_24h) to capture short-term and daily trends.

        - Creates the target (target) as the next hour’s ride_count.

        - Drops rows with missing values due to lagging/shifting (e.g., first 24 hours per zone due to lag_24_ride_count), resulting in ~8,349 rows.

    2. Data Split: Splits the data into training (2019-03-24 23:00:00 to 2019-03-30 23:00:00, 144 hours) and test (2019-03-31 00:00:00 to 23:00:00, 24 hours) sets.

    3. Evaluation with Time-Series Cross-Validation:
        - Uses TimeSeriesSplit with 5 folds to evaluate the model on the training set, ensuring temporal order is preserved.

        - Computes MAE for each fold, then calculates the mean and standard deviation of the cross-validation MAE scores.

        - Also computes MAE, MSE, and RMSE on the full training and test sets.

    4. Two-Step Hyperparameter Tuning:
        - Step 1: Tunes the learning_rate ([0.3, 0.4, 0.5, 0.6, 0.7]) with a fixed n_estimators=200, selecting the best based on cross-validation MAE.

        - Step 2: Tunes other parameters (n_estimators, max_depth, subsample, min_child_weight) with the best learning_rate, again using cross-validation MAE.

    5. Final Model: Trains the final model on the entire training set with the best hyperparameters and evaluates it on both training and test sets.

    6. Output: Saves the model and metadata to models_and_metadata directory, logs to MLflow, and registers the model in the MLflow Model Registry.

**Challenges**

- Overfitting: The final model (learning_rate=0.05, max_depth=7, min_child_weight=5, n_estimators=200, subsample=0.8) showed significant overfitting: train_mae=16.43 vs. test_mae=38.09. This was likely due to the small training dataset (8,349 rows, 6 days) and the model’s complexity.

- Temporal Dependencies: Standard cross-validation would lead to data leakage in a time-series context. TimeSeriesSplit was used to ensure that validation data always came after training data, mimicking real-world forecasting.

- Hyperparameter Tuning: The two-step tuning process was computationally expensive due to the nested cross-validation. A grid search was used for simplicity, but a more efficient method (e.g., Bayesian optimization) could be considered in production.

- Feature Mismatch: The feature store lacked lagged and rolling features, which were added during training. This required careful alignment to ensure consistency with inference.

**Output**

- Model Metrics (from metadata):
    - Training: train_mae=16.43, train_mse=1197.37, train_rmse=34.60

    - Test: test_mae=38.09, test_mse=5221.39, test_rmse=72.26

    - Cross-Validation: cv_mae_mean=26.62, cv_mae_std=8.03

- Training Data Size: 8,349 rows

- Feature Count: 17

- Model File: xgboost_20250407_234708.joblib

- Metadata File: xgboost_metadata_20250407_234708.json

---

### 4. Inference (`03_inference_pipeline.py`)

**Purpose**

The 03_inference_pipeline.py script generates predictions for a specified future timestamp (2019-04-01 00:00:00) using the trained XGBoost model, preparing features for all Manhattan zones and saving the predictions for operational use.

**Reasoning and Logic**

- Input Data: The feature store (features.parquet) and the trained model/metadata from the MLflow Model Registry.

- Steps:
    1. Feature Preparation:
        - Loads the feature store and verifies the latest timestamp (2019-03-31 23:00:00).

        - Creates a DataFrame for the target timestamp (2019-04-01 00:00:00) for all 69 Manhattan zones.

        - Adds all features required by the model (as per feature_columns in metadata):
            - Temporal: hour_of_day, day_of_week, day_of_month, is_weekend, is_holiday, is_rush_hour

            - Weather: temperature, humidity, wind_speed, cloud_cover_numeric, amount_of_precipitation (hardcoded for now)

            - Zone: is_downtown

            - Lagged: lag_1_ride_count, lag_2_ride_count, lag_24_ride_count (computed from historical ride_count)

            - Rolling: rolling_mean_3h, rolling_mean_24h (computed from historical ride_count)

    2. Model Loading: Loads the latest model version from the MLflow Model Registry and the corresponding metadata.

    3. Prediction: Uses the model to predict ride_count for each zone, rounding predictions to the nearest integer.

    4. Output: Saves predictions (hourly_timestamp, PULocationID, predicted_ride_count) to a Parquet file (predictions_20190401_000000.parquet).

**Challenges**

- Feature Alignment: Ensuring that the features prepared for inference exactly matched those used during training was critical. Missing features (e.g., rolling_mean_3h, rolling_mean_24h) were initially overlooked but later added.

- Weather Data: The inference script uses hardcoded weather values due to the lack of real-time weather data for 2019-04-01 00:00:00. In production, this would need to be replaced with an API call to fetch actual weather data.

- Prediction Quality: The model’s high test_mae (38.09) meant that predictions could be off by 38 rides on average, leading to overestimations in high-demand zones (e.g., Zone 249: 292 rides predicted at midnight, which seems high).

- Scalability: While the inference step is fast for a single hour (69 rows), scaling to multiple hours or zones would require optimization, especially for feature preparation.

**Output**

- Predictions File: predictions_20190401_000000.parquet with 69 rows:
    - Example: Zone 249 (West Village) predicted 292 rides, Zone 263 (Upper West Side) predicted 200 rides, Zone 12 (Battery Park City) predicted 1 ride.

- Execution Time: ~6 seconds, indicating good performance for a small inference task.

---

## Overall Challenges and Solutions

1. Data Quality and Consistency:
    - Challenge: Missing values, outliers, and inconsistent timestamps in the raw data.

    - Solution: Implemented robust cleaning steps in 00_data_ingestor.py (e.g., imputation, outlier removal, timestamp validation) to ensure high-quality data for modeling.

2. Temporal Dependencies:
    - Challenge: Standard machine learning techniques (e.g., random train-test splits) would lead to data leakage in a time-series context.

    - Solution: Used TimeSeriesSplit for cross-validation and ensured that lagged features were computed correctly to respect temporal order.

3. Model Overfitting:
    - Challenge: The XGBoost model showed significant overfitting (train_mae=16.43 vs. test_mae=38.09), likely due to the small training dataset and model complexity.

    - Solution: Employed cross-validation to get a more realistic estimate of performance (cv_mae_mean=26.62) and used regularization parameters (min_child_weight, subsample). However, further improvements (e.g., more data, stronger regularization) are needed.

4. Feature Engineering for Inference:
    - Challenge: The inference pipeline needed to replicate the exact feature set used during training, including lagged and rolling features.

    - Solution: Carefully computed all features in 03_inference_pipeline.py, ensuring alignment with the training pipeline (e.g., adding rolling_mean_3h and rolling_mean_24h).

5. Scalability and Production Readiness:
    - Challenge: The pipeline needed to handle large datasets (e.g., millions of rows in 00_data_ingestor.py) and be production-ready for real-time inference.

    - Solution: Used Dask for scalable data processing, MLflow for model management, and designed the inference pipeline to be modular and extensible (e.g., for dynamic timestamps or real-time weather data).

---

## Conclusion

The Batch-Scoring System successfully predicts hourly taxi demand in Manhattan zones, processing raw data, engineering features, training a model, and generating predictions for 2019-04-01 00:00:00. However, the model’s performance (test_mae=38.09) indicates room for improvement, particularly in reducing overfitting and improving prediction accuracy for low-demand periods (e.g., midnight).

**Key Achievements**

- Built a scalable pipeline that processes millions of rows and generates predictions in seconds.

- Incorporated domain knowledge through features like is_rush_hour, is_downtown, and weather data.

- Used time-series cross-validation and two-step hyperparameter tuning to ensure robust model evaluation.

- Successfully predicted demand for 2019-04-01 00:00:00, with predictions that are generally plausible but overestimate in high-demand zones.

This Batch-Scoring System provides a solid foundation for taxi demand forecasting, with clear paths for further improvement to enhance accuracy and operational utility.


