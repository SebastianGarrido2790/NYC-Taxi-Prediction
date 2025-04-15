import pandas as pd
from pathlib import Path
import logging
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import json
import holidays
import requests
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_STORE_DIR = BASE_DIR / "models" / "batch-scoring_system" / "feature_store"
MODELS_DIR = BASE_DIR / "models" / "batch-scoring_system" / "models_and_metadata"
PREDICTIONS_DIR = BASE_DIR / "models" / "batch-scoring_system" / "predictions"
DATA_EXTERNAL_DIR = BASE_DIR / "data" / "external"

# Ensure predictions directory exists
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow Model Registry setup
REGISTERED_MODEL_NAME = "NYC_Taxi_Demand_XGBoost"
client = MlflowClient()

# Manhattan zones (consistent with previous pipelines)
MANHATTAN_ZONES = [
    4,
    12,
    13,
    24,
    41,
    42,
    43,
    45,
    48,
    50,
    68,
    74,
    75,
    79,
    87,
    88,
    90,
    100,
    103,
    104,
    105,
    107,
    113,
    114,
    116,
    120,
    125,
    127,
    128,
    137,
    140,
    141,
    142,
    143,
    144,
    148,
    151,
    152,
    153,
    158,
    161,
    162,
    163,
    164,
    166,
    170,
    186,
    194,
    202,
    209,
    211,
    224,
    229,
    230,
    231,
    232,
    233,
    234,
    236,
    237,
    238,
    239,
    243,
    244,
    246,
    249,
    261,
    262,
    263,
]


def fetch_real_time_weather(timestamp):
    """Fetch real-time weather data for NYC at the given timestamp using OpenWeatherMap API."""
    logger.info(f"Fetching real-time weather data for {timestamp}...")
    try:
        api_key = "your_openweathermap_api_key"  # Replace with your actual OpenWeatherMap API key
        lat, lon = 40.7128, -74.0060  # NYC coordinates
        dt = int(timestamp.timestamp())
        url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={dt}&appid={api_key}&units=imperial"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        weather = {
            "hourly_timestamp": pd.to_datetime(timestamp),
            "temperature": data["current"]["temp"],
            "humidity": data["current"]["humidity"],
            "wind_speed": data["current"]["wind_speed"],
            "cloud_cover_numeric": data["current"]["clouds"],
            "amount_of_precipitation": data["current"].get("rain", {}).get("1h", 0.0),
        }
        weather_df = pd.DataFrame([weather])
        weather_df["hourly_timestamp"] = weather_df["hourly_timestamp"].astype(
            "datetime64[us]"
        )
        logger.info(f"Successfully fetched real-time weather data for {timestamp}")
        return weather_df
    except Exception as e:
        logger.error(f"Error fetching real-time weather data: {e}")
        raise


def preprocess_weather_data(
    historical_only=True, start_date=None, end_date=None, df=None
):
    """Load and preprocess the NYC weather dataset, or fetch real-time data with fallback."""
    logger.info("Loading and preprocessing NYC weather data...")

    if historical_only:
        weather_file = DATA_EXTERNAL_DIR / "nyc_weather.csv"
        if not weather_file.exists():
            raise FileNotFoundError(f"Weather data file {weather_file} not found.")
        weather_df = pd.read_csv(weather_file)

        weather_df["date and time"] = pd.to_datetime(
            weather_df["date and time"], format="%d.%m.%Y %H:%M"
        )
        weather_df["hourly_timestamp"] = weather_df["date and time"].dt.round("h")
        weather_df["hourly_timestamp"] = weather_df["hourly_timestamp"].astype(
            "datetime64[us]"
        )

        weather_df["amount of precipitation"] = weather_df[
            "amount of precipitation"
        ].replace("Trace of precipitation", 0.1)
        weather_df["amount of precipitation"] = pd.to_numeric(
            weather_df["amount of precipitation"], errors="coerce"
        ).fillna(0)

        cloud_cover_mapping = {
            "no clouds": 0,
            "20–30%.": 25,
            "50%.": 50,
            "70 – 80%.": 75,
            "100%.": 100,
            "Sky obscured by fog and/or other meteorological phenomena.": 100,
        }
        weather_df["cloud_cover_numeric"] = (
            weather_df["cloud cover"].map(cloud_cover_mapping).fillna(75)
        )

        weather_df = weather_df.rename(
            columns={
                "wind speed": "wind_speed",
                "amount of precipitation": "amount_of_precipitation",
            }
        )

        weather_df = weather_df[
            [
                "hourly_timestamp",
                "temperature",
                "humidity",
                "wind_speed",
                "cloud_cover_numeric",
                "amount_of_precipitation",
            ]
        ]

        weather_df.set_index("hourly_timestamp", inplace=True)
        weather_df = weather_df.resample("h").ffill().reset_index()
        weather_df["hourly_timestamp"] = weather_df["hourly_timestamp"].astype(
            "datetime64[us]"
        )

        # Filter for the specified date range
        if start_date and end_date:
            weather_df = weather_df[
                (weather_df["hourly_timestamp"] >= start_date)
                & (weather_df["hourly_timestamp"] <= end_date)
            ]

        logger.info(f"Preprocessed weather data: {weather_df.shape[0]} rows")
        return weather_df
    else:
        # Fetch real-time weather data for the latest timestamp with fallback
        latest_timestamp = end_date
        try:
            weather_df = fetch_real_time_weather(latest_timestamp)
        except Exception as e:
            logger.warning(
                f"Failed to fetch real-time weather: {e}. Using last known weather data."
            )
            if df is None:
                raise ValueError(
                    "Feature DataFrame 'df' must be provided for fallback weather data."
                )
            last_weather = (
                df[df["hourly_timestamp"] == df["hourly_timestamp"].max()][
                    [
                        "temperature",
                        "humidity",
                        "wind_speed",
                        "cloud_cover_numeric",
                        "amount_of_precipitation",
                    ]
                ]
                .mean()
                .to_dict()
            )
            weather_df = pd.DataFrame(
                [{"hourly_timestamp": latest_timestamp, **last_weather}]
            )
        return weather_df


def load_latest_features():
    """Load the latest features from the feature store."""
    logger.info("Loading latest features for inference...")

    feature_file = FEATURE_STORE_DIR / "features.parquet"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file {feature_file} not found.")

    df = pd.read_parquet(feature_file)
    df["hourly_timestamp"] = pd.to_datetime(df["hourly_timestamp"])
    logger.info(f"Loaded features with {len(df)} rows.")
    return df


def get_latest_model_and_metadata(model_name="xgboost"):
    """Load the latest model and metadata files based on timestamps."""
    logger.info(f"Loading latest model and metadata for {model_name}...")

    # Find the latest model and metadata files
    model_files = list(MODELS_DIR.glob(f"{model_name}_*.joblib"))
    metadata_files = list(MODELS_DIR.glob(f"{model_name}_metadata_*.json"))

    if not model_files or not metadata_files:
        raise FileNotFoundError("No model or metadata files found.")

    latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
    latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)

    # Load the model
    model = joblib.load(latest_model_file)

    # Load the metadata
    with open(latest_metadata_file, "r") as f:
        metadata = json.load(f)

    logger.info(
        f"Loaded model from {latest_model_file} and metadata from {latest_metadata_file}."
    )
    return model, metadata


def prepare_next_hour_features(df, manhattan_zones, target_timestamp):
    """Prepare features for the target timestamp (2019-04-01 00:00:00)."""
    logger.info(f"Preparing features for {target_timestamp}...")

    # Verify the latest timestamp in the Feature Store
    latest_timestamp = df["hourly_timestamp"].max()
    if latest_timestamp != pd.to_datetime("2019-03-31 23:00:00"):
        raise ValueError(
            f"Expected latest timestamp to be 2019-03-31 23:00:00, but found {latest_timestamp}"
        )

    # Create a DataFrame for the target timestamp for all Manhattan zones
    next_hour_df = pd.DataFrame(
        {
            "hourly_timestamp": [target_timestamp] * len(manhattan_zones),
            "PULocationID": manhattan_zones,
        }
    )

    # Add temporal features
    next_hour_df["hour_of_day"] = next_hour_df["hourly_timestamp"].dt.hour
    next_hour_df["day_of_week"] = next_hour_df["hourly_timestamp"].dt.dayofweek
    next_hour_df["day_of_month"] = next_hour_df["hourly_timestamp"].dt.day
    next_hour_df["is_weekend"] = next_hour_df["day_of_week"].isin([5, 6]).astype("int8")
    next_hour_df["is_rush_hour"] = (
        (next_hour_df["hour_of_day"].isin([7, 8, 9, 16, 17, 18, 19]))
        & (next_hour_df["is_weekend"] == 0)
    ).astype("int8")

    # Add holiday feature
    us_holidays = holidays.US(years=2019)
    next_hour_df["is_holiday"] = (
        next_hour_df["hourly_timestamp"].dt.date.isin(us_holidays).astype("int8")
    )

    # Add real-time weather data for the target timestamp
    weather_df = preprocess_weather_data(
        historical_only=False,
        start_date=target_timestamp,
        end_date=target_timestamp,
        df=df,  # Pass the df parameter for fallback
    )
    next_hour_df = next_hour_df.merge(weather_df, on="hourly_timestamp", how="left")
    next_hour_df["temperature"] = (
        next_hour_df["temperature"].fillna(0).astype("float64")
    )
    next_hour_df["humidity"] = next_hour_df["humidity"].fillna(0).astype("float64")
    next_hour_df["wind_speed"] = next_hour_df["wind_speed"].fillna(0).astype("float64")
    next_hour_df["cloud_cover_numeric"] = (
        next_hour_df["cloud_cover_numeric"].fillna(0).astype("float64")
    )
    next_hour_df["amount_of_precipitation"] = (
        next_hour_df["amount_of_precipitation"].fillna(0).astype("float64")
    )

    # Add zone-specific features
    downtown_zones = [113, 161, 162, 230]  # Lower Manhattan, Midtown, Times Square
    next_hour_df["is_downtown"] = (
        next_hour_df["PULocationID"].isin(downtown_zones).astype("int8")
    )

    # Use precomputed lagged and rolling features from the feature store
    # For lag_1_ride_count (1 hour ago: 2019-03-31 23:00:00)
    lag_1_data = df[
        df["hourly_timestamp"] == pd.to_datetime("2019-03-31 23:00:00")
    ].set_index("PULocationID")
    next_hour_df["lag_1_ride_count"] = (
        next_hour_df["PULocationID"]
        .map(lag_1_data["ride_count"])
        .fillna(0)
        .astype("float64")
    )

    # For lag_2_ride_count (2 hours ago: 2019-03-31 22:00:00)
    lag_2_data = df[
        df["hourly_timestamp"] == pd.to_datetime("2019-03-31 22:00:00")
    ].set_index("PULocationID")
    next_hour_df["lag_2_ride_count"] = (
        next_hour_df["PULocationID"]
        .map(lag_2_data["ride_count"])
        .fillna(0)
        .astype("float64")
    )

    # For lag_24_ride_count (24 hours ago: 2019-03-31 00:00:00)
    lag_24_data = df[
        df["hourly_timestamp"] == pd.to_datetime("2019-03-31 00:00:00")
    ].set_index("PULocationID")
    next_hour_df["lag_24_ride_count"] = (
        next_hour_df["PULocationID"]
        .map(lag_24_data["ride_count"])
        .fillna(0)
        .astype("float64")
    )

    # For rolling_mean_3h and rolling_mean_24h, fetch precomputed values from the feature store
    # rolling_mean_3h at 2019-03-31 23:00:00 (computed from 21:00:00 to 23:00:00)
    rolling_3h_data = df[
        df["hourly_timestamp"] == pd.to_datetime("2019-03-31 23:00:00")
    ].set_index("PULocationID")
    next_hour_df["rolling_mean_3h"] = (
        next_hour_df["PULocationID"]
        .map(rolling_3h_data["rolling_mean_3h"])
        .fillna(0)
        .astype("float64")
    )

    # rolling_mean_24h at 2019-03-31 23:00:00 (computed from 2019-03-31 00:00:00 to 23:00:00)
    next_hour_df["rolling_mean_24h"] = (
        next_hour_df["PULocationID"]
        .map(rolling_3h_data["rolling_mean_24h"])
        .fillna(0)
        .astype("float64")
    )

    logger.info(
        f"Prepared features for {target_timestamp} with {len(next_hour_df)} rows."
    )
    return next_hour_df


def make_predictions(df, model, metadata):
    """Make predictions using the trained model."""
    logger.info("Making predictions...")

    feature_columns = metadata["feature_columns"]
    X = df[feature_columns]

    # Ensure no missing features
    if X.isnull().any().any():
        raise ValueError("Missing values in features for inference.")

    predictions = model.predict(X)
    df["predicted_ride_count"] = predictions.round().astype(
        int
    )  # Round to nearest integer

    logger.info(f"Generated predictions for {len(df)} rows.")
    return df


def save_predictions(df):
    """Save the predictions to the predictions directory."""
    output_file = (
        PREDICTIONS_DIR
        / f"predictions_{df['hourly_timestamp'].iloc[0].strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    logger.info(f"Saving predictions to {output_file}...")

    df[["hourly_timestamp", "PULocationID", "predicted_ride_count"]].to_parquet(
        output_file, engine="pyarrow", compression="snappy", index=False
    )
    logger.info(f"Saved predictions to {output_file}")


def main():
    logger.info("Starting inference pipeline...")

    # Load the latest features
    df = load_latest_features()

    # Define the target timestamp for prediction
    target_timestamp = pd.to_datetime("2019-04-01 00:00:00")

    # Prepare features for the target timestamp
    next_hour_df = prepare_next_hour_features(df, MANHATTAN_ZONES, target_timestamp)

    # Load the latest model and metadata dynamically
    model, metadata = get_latest_model_and_metadata()

    # Make predictions
    predictions_df = make_predictions(next_hour_df, model, metadata)

    # Save predictions
    save_predictions(predictions_df)

    logger.info("Inference pipeline completed.")


if __name__ == "__main__":
    main()
