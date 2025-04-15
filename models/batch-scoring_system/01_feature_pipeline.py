import pandas as pd
import numpy as np
from pathlib import Path
import logging
import holidays
import pyarrow as pa
from datetime import datetime, timedelta
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim" / "production"
DATA_EXTERNAL_DIR = BASE_DIR / "data" / "external"
FEATURE_STORE_DIR = BASE_DIR / "models" / "batch-scoring_system" / "feature_store"

# Ensure feature store directory exists
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Manhattan zones
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
        api_key = "your_openweathermap_api_key"  # Replace with your API key
        lat, lon = 40.7128, -74.0060  # NYC coordinates
        dt = int(timestamp.timestamp())
        url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={dt}&appid={api_key}&units=imperial"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
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


def preprocess_weather_data(historical_only=True, start_date=None, end_date=None):
    """Load and preprocess the NYC weather dataset, or fetch real-time data."""
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
        # Fetch real-time weather data for the latest timestamp (historical_only=False)
        latest_timestamp = end_date
        return fetch_real_time_weather(latest_timestamp)


def aggregate_ride_counts(df):
    """Aggregate the data by hourly_timestamp and PULocationID to compute ride_count."""
    logger.info("Aggregating data to compute ride counts...")

    df["hourly_timestamp"] = df["tpep_pickup_datetime"].dt.round("h")
    df["hourly_timestamp"] = df["hourly_timestamp"].astype("datetime64[us]")

    ride_counts = (
        df.groupby(["hourly_timestamp", "PULocationID"])
        .size()
        .reset_index(name="ride_count")
    )
    ride_counts["ride_count"] = ride_counts["ride_count"].astype("int32")

    logger.info(f"Aggregated ride counts: {len(ride_counts)} rows")
    return ride_counts


def create_grid(ride_counts, start_date, end_date):
    """Create a complete grid of timestamps and Manhattan zones."""
    logger.info("Creating grid for all timestamps and Manhattan zones...")

    # Generate all timestamps in the range
    all_timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    grid = pd.MultiIndex.from_product(
        [all_timestamps, MANHATTAN_ZONES], names=["hourly_timestamp", "PULocationID"]
    ).to_frame(index=False)
    grid["hourly_timestamp"] = grid["hourly_timestamp"].astype("datetime64[us]")
    grid["PULocationID"] = grid["PULocationID"].astype("int32")

    # Merge with ride counts, filling missing with 0
    ride_counts = grid.merge(
        ride_counts, how="left", on=["hourly_timestamp", "PULocationID"]
    ).fillna({"ride_count": 0})
    ride_counts["ride_count"] = ride_counts["ride_count"].astype("int32")

    logger.info(f"Grid created with {len(ride_counts)} rows")
    return ride_counts


def add_temporal_features(df):
    """Add temporal features to the DataFrame."""
    logger.info("Adding temporal features...")

    df["hour_of_day"] = df["hourly_timestamp"].dt.hour
    df["day_of_week"] = df["hourly_timestamp"].dt.dayofweek
    df["day_of_month"] = df["hourly_timestamp"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["is_rush_hour"] = (
        (df["hour_of_day"].isin([7, 8, 9, 16, 17, 18, 19])) & (df["is_weekend"] == 0)
    ).astype("int8")

    us_holidays = holidays.US(years=2019)
    df["is_holiday"] = df["hourly_timestamp"].dt.date.isin(us_holidays).astype("int8")

    return df


def load_taxi_zone_lookup():
    """Load the Taxi Zone Lookup Table."""
    lookup_file = DATA_EXTERNAL_DIR / "taxi_zone_lookup.csv"
    if not lookup_file.exists():
        raise FileNotFoundError(f"Taxi Zone Lookup file not found at {lookup_file}")
    return pd.read_csv(lookup_file)[["LocationID", "Zone"]]


def add_zone_features(df):
    """Add zone-specific features from the Taxi Zone Lookup Table."""
    logger.info("Adding zone-specific features...")
    lookup_df = load_taxi_zone_lookup()
    df = df.merge(lookup_df, how="left", left_on="PULocationID", right_on="LocationID")
    downtown_zones = [113, 161, 162, 230]  # Lower Manhattan, Midtown, Times Square
    df["is_downtown"] = df["PULocationID"].isin(downtown_zones).astype("int8")
    return df


def merge_weather_data(df, weather_df):
    """Merge the trip data with weather data."""
    logger.info("Merging weather data...")

    df = df.merge(weather_df, how="left", on="hourly_timestamp")
    df["temperature"] = df["temperature"].ffill().fillna(0)
    df["humidity"] = df["humidity"].ffill().fillna(0)
    df["wind_speed"] = df["wind_speed"].ffill().fillna(0)
    df["cloud_cover_numeric"] = df["cloud_cover_numeric"].ffill().fillna(0)
    df["amount_of_precipitation"] = df["amount_of_precipitation"].fillna(0)

    return df


def add_lagged_and_rolling_features(df):
    """Add lagged and rolling features to the DataFrame."""
    logger.info("Adding lagged and rolling features...")
    df = df.sort_values(["PULocationID", "hourly_timestamp"])
    # Add lagged features
    df["lag_1_ride_count"] = df.groupby("PULocationID")["ride_count"].shift(1)
    df["lag_2_ride_count"] = df.groupby("PULocationID")["ride_count"].shift(2)
    df["lag_24_ride_count"] = df.groupby("PULocationID")["ride_count"].shift(24)
    # Add rolling mean features
    df["rolling_mean_3h"] = (
        df.groupby("PULocationID")["ride_count"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_mean_24h"] = (
        df.groupby("PULocationID")["ride_count"]
        .rolling(window=24, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # Fill NaN values with 0
    df = df.fillna(0)
    # Ensure correct data types
    df["lag_1_ride_count"] = df["lag_1_ride_count"].astype("float64")
    df["lag_2_ride_count"] = df["lag_2_ride_count"].astype("float64")
    df["lag_24_ride_count"] = df["lag_24_ride_count"].astype("float64")
    df["rolling_mean_3h"] = df["rolling_mean_3h"].astype("float64")
    df["rolling_mean_24h"] = df["rolling_mean_24h"].astype("float64")
    return df


def process_features(input_files, start_date, end_date, historical_weather=True):
    """Process raw data into features for the given time range from multiple files."""
    logger.info(f"Processing features for data in {input_files}...")

    # Load and concatenate all input files
    dfs = []
    for input_file in input_files:
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded data from {input_file}: {len(df)} rows")
        dfs.append(df)

    if not dfs:
        raise ValueError("No data files to process.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined raw data: {len(df)} rows")

    # Filter data within the time range
    df = df[
        (df["tpep_pickup_datetime"] >= start_date)
        & (df["tpep_pickup_datetime"] <= end_date)
    ]
    if len(df) == 0:
        logger.warning("No data within the specified time range.")
        return pd.DataFrame()

    # Step 1: Aggregate to compute ride counts
    ride_counts = aggregate_ride_counts(df)

    # Step 2: Create a complete grid
    ride_counts = create_grid(ride_counts, start_date, end_date)

    # Step 3: Add temporal features
    df = add_temporal_features(ride_counts)

    # Step 4: Add zone-specific features
    df = add_zone_features(df)

    # Step 5: Load and preprocess weather data
    weather_df = preprocess_weather_data(
        historical_only=historical_weather, start_date=start_date, end_date=end_date
    )

    # Step 6: Merge with weather data
    df = merge_weather_data(df, weather_df)

    # Step 7: Add lagged and rolling features
    df = add_lagged_and_rolling_features(df)

    # Step 8: Add zone as a categorical feature
    df["zone"] = df["PULocationID"].astype("int32")

    return df


def save_features(df, output_file, append=False):
    """Save the features to the feature store, with option to append."""
    logger.info(f"Saving features to {output_file}...")

    schema = pa.schema(
        [
            ("hourly_timestamp", pa.timestamp("us")),
            ("PULocationID", pa.int32()),
            ("ride_count", pa.int32()),
            ("hour_of_day", pa.int32()),
            ("day_of_week", pa.int32()),
            ("day_of_month", pa.int32()),
            ("is_weekend", pa.int8()),
            ("is_holiday", pa.int8()),
            ("is_rush_hour", pa.int8()),
            ("temperature", pa.float64()),
            ("humidity", pa.float64()),
            ("wind_speed", pa.float64()),
            ("cloud_cover_numeric", pa.float64()),
            ("amount_of_precipitation", pa.float64()),
            ("is_downtown", pa.int8()),
            ("lag_1_ride_count", pa.float64()),
            ("lag_2_ride_count", pa.float64()),
            ("lag_24_ride_count", pa.float64()),
            ("rolling_mean_3h", pa.float64()),
            ("rolling_mean_24h", pa.float64()),
            ("zone", pa.int32()),
            ("LocationID", pa.int32()),
            ("Zone", pa.string()),
        ]
    )

    if append and output_file.exists():
        existing_df = pd.read_parquet(output_file)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(
            subset=["hourly_timestamp", "PULocationID"], keep="last"
        )
        combined_df.to_parquet(
            output_file, engine="pyarrow", compression="snappy", schema=schema
        )
    else:
        df.to_parquet(
            output_file, engine="pyarrow", compression="snappy", schema=schema
        )

    logger.info(f"Saved features to {output_file}")


def get_all_data_files():
    """Get all cleaned data files in the interim directory."""
    data_files = list(DATA_INTERIM_DIR.glob("cleaned_yellow_tripdata_*.parquet"))
    if not data_files:
        raise FileNotFoundError("No cleaned data files found in interim directory.")
    return sorted(data_files, key=lambda x: x.stat().st_mtime)


def main():
    logger.info("Starting feature pipeline...")

    # Get all data files
    input_files = get_all_data_files()
    logger.info(f"Found {len(input_files)} data files: {[f.name for f in input_files]}")

    # Define the time range (last 7 days for more data)
    end_date = pd.to_datetime("2019-03-31 23:00:00")
    start_date = end_date - timedelta(days=7)  # From 2019-03-25 to 2019-03-31
    logger.info(f"Processing data from {start_date} to {end_date}")

    # Process features from all files
    features_df = process_features(
        input_files,
        start_date,
        end_date,
        historical_weather=True,  # Set to False to use real-time API
    )

    # Save to feature store (append mode to support incremental updates)
    output_file = FEATURE_STORE_DIR / "features.parquet"
    if not features_df.empty:
        save_features(features_df, output_file, append=True)

    logger.info("Feature pipeline completed.")


if __name__ == "__main__":
    main()
