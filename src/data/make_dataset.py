import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pyarrow as pa

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Ensure processed directory exists
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Manhattan zones (PULocationIDs for Manhattan)
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


def load_featured_data():
    """Load the featured data for all months using Pandas."""
    logger.info("Loading featured data...")

    # List of months to load
    months = ["january", "february", "march"]
    dfs = []

    # Define the time range
    start_date = pd.to_datetime("2019-01-01 00:00:00")
    end_date = pd.to_datetime("2019-03-31 23:00:00")

    for month in months:
        file_path = DATA_INTERIM_DIR / f"featured_yellow_tripdata_2019-{month}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Featured data file {file_path} not found.")
        df = pd.read_parquet(file_path)
        # Ensure hourly_timestamp is datetime64[us]
        df["hourly_timestamp"] = pd.to_datetime(df["hourly_timestamp"])
        # Filter timestamps within the desired range
        df = df[
            (df["hourly_timestamp"] >= start_date)
            & (df["hourly_timestamp"] <= end_date)
        ]
        # Convert zone to category
        df["zone"] = df["zone"].astype("category")
        # Remove duplicates based on hourly_timestamp and PULocationID
        initial_rows = len(df)
        df = df.drop_duplicates(subset=["hourly_timestamp", "PULocationID"])
        logger.info(f"Dropped {initial_rows - len(df)} duplicates for {month}.")
        dfs.append(df)

    # Concatenate all months into a single Pandas DataFrame
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    num_rows = len(combined_df)
    logger.info(f"Loaded featured data with {num_rows} rows (before aggregation).")

    # Validate that all Manhattan zones are present
    unique_zones = combined_df["PULocationID"].unique()
    missing_zones = set(MANHATTAN_ZONES) - set(unique_zones)
    if missing_zones:
        logger.warning(f"Missing Manhattan zones: {missing_zones}")
    else:
        logger.info("All expected Manhattan zones are present.")

    return combined_df


def aggregate_to_time_series(df):
    """Aggregate the data into hourly counts per zone and add lagged features using Pandas."""
    logger.info("Aggregating data into time-series...")

    # Step 1: Use the existing ride_count column (no need to recompute)
    ride_counts = df[["hourly_timestamp", "PULocationID", "ride_count"]].copy()
    # Ensure no duplicates
    ride_counts = ride_counts.drop_duplicates(
        subset=["hourly_timestamp", "PULocationID"]
    )
    num_rows = len(ride_counts)
    logger.info(f"Ride counts (using existing ride_count): {num_rows} rows")

    # Step 2: Get unique timestamps and zones to create a complete grid
    all_timestamps = pd.date_range(
        start="2019-01-01 00:00:00", end="2019-03-31 23:00:00", freq="h"
    )
    all_zones = df["PULocationID"].unique()
    # Filter to Manhattan zones
    all_zones = all_zones[np.isin(all_zones, MANHATTAN_ZONES)]
    logger.info(f"Number of unique zones (Manhattan only): {len(all_zones)}")
    logger.info(f"Number of timestamps in grid: {len(all_timestamps)}")
    grid = pd.MultiIndex.from_product(
        [all_timestamps, all_zones], names=["hourly_timestamp", "PULocationID"]
    ).to_frame(index=False)
    # Convert hourly_timestamp to datetime64[us] to match the trip data
    grid["hourly_timestamp"] = pd.to_datetime(grid["hourly_timestamp"])
    num_rows = len(grid)
    logger.info(f"Grid size: {num_rows} rows")

    # Step 3: Merge the grid with ride counts to fill missing hours with 0
    initial_rows = len(ride_counts)
    ride_counts = grid.merge(
        ride_counts, how="left", on=["hourly_timestamp", "PULocationID"]
    ).fillna({"ride_count": 0})
    num_rows = len(ride_counts)
    filled_rows = num_rows - initial_rows
    logger.info(f"Filled {filled_rows} missing hours/zones with 0 ride counts.")
    logger.info(f"After merging with grid: {num_rows} rows")

    # Step 4: Merge back with the features (exclude ride_count from features to avoid conflict)
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
        "zone",
    ]
    features = df[["hourly_timestamp", "PULocationID"] + feature_columns].copy()
    features = (
        features.groupby(["hourly_timestamp", "PULocationID"]).first().reset_index()
    )
    # Ensure no duplicates in features
    features = features.drop_duplicates(subset=["hourly_timestamp", "PULocationID"])
    num_rows = len(features)
    logger.info(f"Features size: {num_rows} rows")
    time_series_df = ride_counts.merge(
        features, how="left", on=["hourly_timestamp", "PULocationID"]
    )
    num_rows = len(time_series_df)
    logger.info(f"After merging with features: {num_rows} rows")

    # Step 4.1: Fill missing feature values
    # Log missing features before filling
    missing_features = time_series_df[feature_columns].isna().mean() * 100
    logger.info(f"Missing feature percentages before filling:\n{missing_features}")
    # Sort by PULocationID and hourly_timestamp
    time_series_df = time_series_df.sort_values(["PULocationID", "hourly_timestamp"])
    # Forward fill within each PULocationID group
    time_series_df[feature_columns] = time_series_df.groupby("PULocationID")[
        feature_columns
    ].ffill()
    # Backward fill to handle any remaining NaNs at the start
    time_series_df[feature_columns] = time_series_df.groupby("PULocationID")[
        feature_columns
    ].bfill()
    # For any remaining NaNs (should be rare), fill with reasonable defaults
    time_series_df["hour_of_day"] = time_series_df["hourly_timestamp"].dt.hour
    time_series_df["day_of_week"] = time_series_df["hourly_timestamp"].dt.dayofweek
    time_series_df["day_of_month"] = time_series_df["hourly_timestamp"].dt.day
    time_series_df["is_weekend"] = (
        time_series_df["day_of_week"].isin([5, 6]).astype("int8")
    )
    time_series_df["is_holiday"] = time_series_df["is_holiday"].fillna(0).astype("int8")
    time_series_df["is_rush_hour"] = (
        (time_series_df["hour_of_day"].isin([7, 8, 9, 16, 17, 18, 19]))
        & (time_series_df["is_weekend"] == 0)
    ).astype("int8")
    time_series_df["is_downtown"] = (
        time_series_df["is_downtown"].fillna(0).astype("int8")
    )
    time_series_df[
        [
            "temperature",
            "humidity",
            "wind_speed",
            "cloud_cover_numeric",
            "amount_of_precipitation",
        ]
    ] = time_series_df[
        [
            "temperature",
            "humidity",
            "wind_speed",
            "cloud_cover_numeric",
            "amount_of_precipitation",
        ]
    ].fillna(
        0
    )
    # For 'zone', fill with PULocationID as a fallback (though it shouldn't be NaN)
    time_series_df["zone"] = (
        time_series_df["zone"].fillna(time_series_df["PULocationID"]).astype("category")
    )
    num_rows = len(time_series_df)
    logger.info(f"After filling missing features: {num_rows} rows")
    # Log missing features after filling
    missing_features_after = time_series_df[feature_columns].isna().mean() * 100
    logger.info(f"Missing feature percentages after filling:\n{missing_features_after}")

    # Step 5: Add lagged features (previous hours' ride counts)
    for lag in [1, 2, 3, 6, 12, 24]:  # Lags for 1, 2, 3, 6, 12, 24 hours ago
        time_series_df[f"lag_{lag}_ride_count"] = time_series_df.groupby(
            "PULocationID"
        )["ride_count"].shift(lag)
        time_series_df[f"lag_{lag}_ride_count"] = time_series_df[
            f"lag_{lag}_ride_count"
        ].astype("float64")
    # Add rolling mean features
    time_series_df["rolling_mean_3h"] = (
        time_series_df.groupby("PULocationID")["ride_count"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )
    time_series_df["rolling_mean_24h"] = (
        time_series_df.groupby("PULocationID")["ride_count"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )
    num_rows = len(time_series_df)
    logger.info(f"After adding lagged and rolling features: {num_rows} rows")
    # Log distribution of lagged and rolling features
    lag_columns = [f"lag_{lag}_ride_count" for lag in [1, 2, 3, 6, 12, 24]] + [
        "rolling_mean_3h",
        "rolling_mean_24h",
    ]
    lag_stats = time_series_df[lag_columns].describe()
    logger.info(f"Lagged and rolling feature distribution:\n{lag_stats}")

    # Step 6: Create the target variable (next hour's ride count)
    time_series_df["target"] = time_series_df.groupby("PULocationID")[
        "ride_count"
    ].shift(-1)
    time_series_df["target"] = time_series_df["target"].astype("float64")
    num_rows = len(time_series_df)
    logger.info(f"After adding target: {num_rows} rows")
    # Log target distribution
    target_stats = time_series_df["target"].describe()
    logger.info(f"Target distribution:\n{target_stats}")

    # Step 7: Drop rows with missing target or lagged features
    drop_columns = (
        ["target"]
        + [f"lag_{lag}_ride_count" for lag in [1, 2, 3, 6, 12, 24]]
        + [
            "rolling_mean_3h",
            "rolling_mean_24h",
        ]
    )
    initial_rows = len(time_series_df)
    time_series_df = time_series_df.dropna(subset=drop_columns)
    num_rows = len(time_series_df)
    logger.info(f"Dropped {initial_rows - num_rows} rows with missing values.")
    logger.info(f"After dropping missing values: {num_rows} rows")

    logger.info(f"Aggregated time-series data with {num_rows} rows.")
    return time_series_df


def train_test_split(df):
    """Split the data into training and testing sets."""
    logger.info("Splitting data into train and test sets...")

    # Define the split point: January-February for training, March for testing
    train_end_date = pd.to_datetime("2019-02-28 23:00:00")
    test_start_date = pd.to_datetime("2019-03-01 00:00:00")

    # Split the data
    train_df = df[df["hourly_timestamp"] <= train_end_date]
    test_df = df[df["hourly_timestamp"] >= test_start_date]

    # Validate the split
    train_len = len(train_df)
    test_len = len(test_df)
    if train_len == 0:
        raise ValueError("Training set is empty after splitting.")
    if test_len == 0:
        raise ValueError("Testing set is empty after splitting.")
    logger.info(f"Training set: {train_len} rows")
    logger.info(f"Testing set: {test_len} rows")
    return train_df, test_df


def save_datasets(train_df, test_df):
    """Save the train and test datasets to the processed directory."""
    logger.info("Saving train and test datasets...")

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
            ("lag_1_ride_count", pa.float64()),
            ("lag_2_ride_count", pa.float64()),
            ("lag_3_ride_count", pa.float64()),
            ("lag_6_ride_count", pa.float64()),
            ("lag_12_ride_count", pa.float64()),
            ("lag_24_ride_count", pa.float64()),
            ("rolling_mean_3h", pa.float64()),
            ("rolling_mean_24h", pa.float64()),
            ("target", pa.float64()),
        ]
    )

    # Save the datasets
    train_file = DATA_PROCESSED_DIR / "train_dataset.parquet"
    test_file = DATA_PROCESSED_DIR / "test_dataset.parquet"

    train_df.to_parquet(
        train_file, engine="pyarrow", compression="snappy", schema=schema, index=False
    )
    test_df.to_parquet(
        test_file, engine="pyarrow", compression="snappy", schema=schema, index=False
    )

    logger.info(f"Saved training dataset to {train_file}")
    logger.info(f"Saved testing dataset to {test_file}")


def main():
    logger.info("Starting data processing pipeline...")
    # Load the featured data
    df = load_featured_data()

    # Aggregate into time-series
    time_series_df = aggregate_to_time_series(df)

    # Split into train and test sets
    train_df, test_df = train_test_split(time_series_df)

    # Save the datasets
    save_datasets(train_df, test_df)


if __name__ == "__main__":
    main()
