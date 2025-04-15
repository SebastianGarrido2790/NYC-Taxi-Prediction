**Instead of running inference in the dashboard, precompute the test set predictions and save them to a file.**

```python
import pandas as pd
from pathlib import Path
import xgboost as xgb

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_STORE_DIR = BASE_DIR / "models" / "batch-scoring_system" / "feature_store"
MODELS_DIR = BASE_DIR / "models" / "batch-scoring_system" / "models_and_metadata"
OUTPUT_DIR = BASE_DIR / "models" / "batch-scoring_system" / "predictions"

# Load feature store
feature_file = FEATURE_STORE_DIR / "features.parquet"
feature_df = pd.read_parquet(feature_file)

# Filter for test set
test_df = feature_df[
    (feature_df["hourly_timestamp"] >= "2019-03-31 00:00:00")
    & (feature_df["hourly_timestamp"] < "2019-04-01 00:00:00")
].copy()

# Load model and predict
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
model_file = MODELS_DIR / "xgboost.joblib"
model = xgb.XGBRegressor()
model.load_model(model_file)
X_test = test_df[feature_columns]
test_df["predicted_ride_count"] = model.predict(X_test).round().astype(int)
test_df["actual_ride_count"] = test_df["ride_count"]

# Save predictions
output_file = OUTPUT_DIR / "test_predictions_20190331.parquet"
test_df[
    ["hourly_timestamp", "PULocationID", "predicted_ride_count", "actual_ride_count"]
].to_parquet(output_file)
print(f"Saved test set predictions to {output_file}")
```

**Then update dashboard.py to load the precomputed predictions instead of running inference.**
**Updated prepare_test_set_data in dashboard.py:**

```python
def prepare_test_set_data(feature_df, metadata):
    try:
        test_predictions_file = PREDICTIONS_DIR / "test_predictions_20190331.parquet"
        test_df = pd.read_parquet(test_predictions_file)
        test_df["hourly_timestamp"] = pd.to_datetime(test_df["hourly_timestamp"])
        return test_df
    except Exception as e:
        st.error(f"Error loading test set predictions: {e}")
        return None
```