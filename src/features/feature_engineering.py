import pandas as pd
import numpy as np
from pathlib import Path
import logging
import holidays
import pyarrow as pa

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim"
DATA_EXTERNAL_DIR = BASE_DIR / "data" / "external"

# Ensure interim directory exists
DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_weather_data():
    """Load and preprocess the NYC weather dataset."""
    logger.info("Loading and preprocessing NYC weather data...")

    # Load the weather data
    weather_file = DATA_EXTERNAL_DIR / "nyc_weather.csv"
    if not weather_file.exists():
        raise FileNotFoundError(f"Weather data file {weather_file} not found.")
    weather_df = pd.read_csv(weather_file)

    # Convert 'date and time' to datetime (format: DD.MM.YYYY HH:MM)
    weather_df["date and time"] = pd.to_datetime(
        weather_df["date and time"], format="%d.%m.%Y %H:%M"
    )

    # Round to the nearest hour for merging with trip data
    weather_df["hourly_timestamp"] = weather_df["date and time"].dt.round("h")

    # Convert hourly_timestamp to datetime64[us] to match trip data
    weather_df["hourly_timestamp"] = weather_df["hourly_timestamp"].astype(
        "datetime64[us]"
    )

    # Validate weather data coverage
    weather_start = weather_df["hourly_timestamp"].min()
    weather_end = weather_df["hourly_timestamp"].max()
    expected_start = pd.to_datetime("2019-01-01 00:00:00")
    expected_end = pd.to_datetime("2019-03-31 23:00:00")
    if weather_start > expected_start or weather_end < expected_end:
        logger.warning(
            f"Weather data range ({weather_start} to {weather_end}) does not fully cover "
            f"the expected range ({expected_start} to {expected_end})."
        )

    # Handle missing values in 'amount of precipitation'
    weather_df["amount of precipitation"] = weather_df[
        "amount of precipitation"
    ].replace("Trace of precipitation", 0.1)
    weather_df["amount of precipitation"] = pd.to_numeric(
        weather_df["amount of precipitation"], errors="coerce"
    ).fillna(0)

    # Handle 'cloud cover'
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

    # Rename columns for clarity
    weather_df = weather_df.rename(
        columns={
            "wind speed": "wind_speed",
            "amount of precipitation": "amount_of_precipitation",
        }
    )

    # Select relevant columns
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

    # Resample to ensure hourly data (fill missing hours with forward fill)
    weather_df.set_index("hourly_timestamp", inplace=True)
    weather_df = weather_df.resample("h").ffill().reset_index()

    # Ensure hourly_timestamp is datetime64[us] after resampling
    weather_df["hourly_timestamp"] = weather_df["hourly_timestamp"].astype(
        "datetime64[us]"
    )

    logger.info(f"Preprocessed weather data: {weather_df.shape[0]} rows")
    return weather_df


def aggregate_ride_counts(df):
    """Aggregate the data by hourly_timestamp and PULocationID to compute ride_count."""
    logger.info("Aggregating data to compute ride counts...")

    # Round pickup datetime to the nearest hour
    df["hourly_timestamp"] = df["tpep_pickup_datetime"].dt.round("h")

    # Ensure hourly_timestamp is datetime64[us]
    df["hourly_timestamp"] = df["hourly_timestamp"].astype("datetime64[us]")

    # Aggregate by hourly_timestamp and PULocationID to get ride counts
    ride_counts = (
        df.groupby(["hourly_timestamp", "PULocationID"])
        .size()
        .reset_index(name="ride_count")
    )

    # Convert ride_count to int
    ride_counts["ride_count"] = ride_counts["ride_count"].astype("int32")

    # Log ride count distribution
    ride_count_stats = ride_counts["ride_count"].describe()
    logger.info(f"Ride count distribution:\n{ride_count_stats}")

    logger.info(f"Aggregated ride counts: {len(ride_counts)} rows")
    return ride_counts


def add_temporal_features(df):
    """Add temporal features to the DataFrame."""
    logger.info("Adding temporal features...")

    # Extract temporal features
    df["hour_of_day"] = df["hourly_timestamp"].dt.hour
    df["day_of_week"] = df["hourly_timestamp"].dt.dayofweek
    df["day_of_month"] = df["hourly_timestamp"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["is_rush_hour"] = (
        (df["hour_of_day"].isin([7, 8, 9, 16, 17, 18, 19])) & (df["is_weekend"] == 0)
    ).astype("int8")

    # Add holiday feature (US holidays in 2019)
    us_holidays = holidays.US(years=2019)
    holiday_dates = set(us_holidays.keys())
    df["is_holiday"] = df["hourly_timestamp"].dt.date.isin(holiday_dates).astype("int8")

    return df


def load_taxi_zone_lookup():
    """Load the Taxi Zone Lookup Table."""
    lookup_file = DATA_EXTERNAL_DIR / "taxi_zone_lookup.csv"
    if not lookup_file.exists():
        raise FileNotFoundError(f"Taxi Zone Lookup file not found at {lookup_file}")
    lookup_df = pd.read_csv(lookup_file)
    return lookup_df[["LocationID", "Zone"]]


def add_zone_features(df):
    """Add zone-specific features from the Taxi Zone Lookup Table."""
    logger.info("Adding zone-specific features...")
    lookup_df = load_taxi_zone_lookup()
    df = df.merge(lookup_df, how="left", left_on="PULocationID", right_on="LocationID")
    # Add a binary feature for downtown zones (e.g., Lower Manhattan, Midtown, Times Square)
    downtown_zones = [113, 161, 162, 230]  # Lower Manhattan, Midtown, Times Square
    df["is_downtown"] = df["PULocationID"].isin(downtown_zones).astype("int8")
    df = df.drop(columns=["LocationID", "Zone"])  # Drop redundant columns
    return df


def merge_weather_data(df, weather_df):
    """Merge the trip data with weather data."""
    logger.info("Merging weather data...")

    # Merge on hourly_timestamp
    df = df.merge(weather_df, how="left", on="hourly_timestamp")

    # Log missing weather data after merge
    missing_weather = (
        df[
            [
                "temperature",
                "humidity",
                "wind_speed",
                "cloud_cover_numeric",
                "amount_of_precipitation",
            ]
        ]
        .isna()
        .mean()
        * 100
    )
    logger.info(f"Missing weather data percentages after merge:\n{missing_weather}")

    # Fill missing weather data using forward fill
    df["temperature"] = df["temperature"].ffill()
    df["humidity"] = df["humidity"].ffill()
    df["wind_speed"] = df["wind_speed"].ffill()
    df["cloud_cover_numeric"] = df["cloud_cover_numeric"].ffill()
    df["amount_of_precipitation"] = df["amount_of_precipitation"].fillna(0)

    return df


def process_month(month_name):
    """Process a single month's data and add features."""
    logger.info(f"Processing {month_name} data for feature engineering...")

    # Load the cleaned data with Pandas
    input_file = (
        DATA_INTERIM_DIR / f"cleaned_yellow_tripdata_2019-{month_name.lower()}.parquet"
    )
    if not input_file.exists():
        raise FileNotFoundError(f"Cleaned data file {input_file} not found.")

    df = pd.read_parquet(input_file)
    logger.info(f"Loaded cleaned {month_name} data: {len(df)} rows")

    # Step 1: Aggregate to compute ride counts
    df = aggregate_ride_counts(df)

    # Step 2: Add temporal features
    df = add_temporal_features(df)

    # Step 3: Add zone-specific features
    df = add_zone_features(df)

    # Step 4: Load and preprocess weather data
    weather_df = preprocess_weather_data()

    # Step 5: Merge with weather data
    df = merge_weather_data(df, weather_df)

    # Step 6: Add zone as a categorical feature
    df["zone"] = df["PULocationID"].astype("category")

    # Define the schema for to_parquet
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
            ("zone", pa.dictionary(pa.int32(), pa.int32())),
        ]
    )

    # Select relevant columns
    columns_to_keep = [
        "hourly_timestamp",
        "PULocationID",
        "ride_count",
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
        "zone",
    ]
    df = df[columns_to_keep]

    # Save the data with features
    output_file = (
        DATA_INTERIM_DIR / f"featured_yellow_tripdata_2019-{month_name.lower()}.parquet"
    )
    df.to_parquet(output_file, engine="pyarrow", compression="snappy", schema=schema)
    logger.info(f"Saved featured {month_name} data to {output_file}")


def main():
    # Process each month's data
    months = ["January", "February", "March"]
    for month in months:
        process_month(month)


if __name__ == "__main__":
    main()
