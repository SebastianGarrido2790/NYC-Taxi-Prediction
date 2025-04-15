import pandas as pd
import numpy as np
from pathlib import Path
import logging

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


def load_featured_data(month_name):
    """Load the featured data for a given month."""
    logger.info(f"Loading featured data for {month_name}...")

    input_file = (
        DATA_INTERIM_DIR / f"featured_yellow_tripdata_2019-{month_name.lower()}.parquet"
    )
    if not input_file.exists():
        raise FileNotFoundError(f"Featured data file {input_file} not found.")

    df = pd.read_parquet(input_file)
    logger.info(f"Loaded featured {month_name} data: {len(df)} rows")

    return df


def add_event_features(df):
    """Add synthetic event features for busy zones (e.g., Times Square)."""
    logger.info("Adding synthetic event features...")

    # Define busy zones (based on error analysis)
    busy_zones = [
        186,
        230,
        48,
        79,
        161,
        142,
        162,
        236,
        237,
        170,
    ]  # Top 10 high-error zones

    # Simulate events in busy zones during high-demand periods (e.g., evening hours on weekends)
    # In a real scenario, this would be replaced with actual event data
    df["has_event"] = 0
    event_condition = (
        (df["PULocationID"].isin(busy_zones))  # Busy zones
        & (
            df["hour_of_day"].isin([18, 19, 20, 21, 22, 23])
        )  # Evening hours (6 PM - 11 PM)
        & (df["is_weekend"] == 1)  # Weekends
    )
    df.loc[event_condition, "has_event"] = 1

    # Log the proportion of data with events
    event_proportion = df["has_event"].mean()
    logger.info(f"Proportion of data with simulated events: {event_proportion:.4f}")

    return df


def add_weather_features(df):
    """Add weather-related features, including is_raining and interaction terms."""
    logger.info("Adding weather-related features...")

    # Add is_raining feature (based on amount_of_precipitation)
    df["is_raining"] = (df["amount_of_precipitation"] > 0).astype("int8")

    # Add interaction term: is_raining * is_downtown
    df["is_raining_downtown"] = (df["is_raining"] * df["is_downtown"]).astype("int8")

    # Log the proportion of data with rain
    rain_proportion = df["is_raining"].mean()
    logger.info(f"Proportion of data with rain: {rain_proportion:.4f}")

    return df


def add_temporal_features(df):
    """Add enhanced temporal features, including is_evening_rush and interaction terms."""
    logger.info("Adding enhanced temporal features...")

    # Add is_evening_rush (4 PM - 7 PM on weekdays)
    df["is_evening_rush"] = (
        (df["hour_of_day"].isin([16, 17, 18, 19])) & (df["is_weekend"] == 0)
    ).astype("int8")

    # Add interaction terms
    # 1. is_weekend * hour_of_day (to capture weekend-specific hourly patterns)
    df["is_weekend_hour"] = df["is_weekend"] * df["hour_of_day"]

    # 2. is_rush_hour * is_downtown (to capture rush hour effects in downtown areas)
    df["is_rush_hour_downtown"] = (df["is_rush_hour"] * df["is_downtown"]).astype(
        "int8"
    )

    # 3. is_downtown * hour_of_day (to capture zone-specific hourly patterns)
    df["is_downtown_hour"] = df["is_downtown"] * df["hour_of_day"]

    return df


def process_month(month_name):
    """Process a single month's data to add new features."""
    logger.info(f"Processing {month_name} data for new feature engineering...")

    # Load the featured data
    df = load_featured_data(month_name)

    # Step 1: Add synthetic event features
    df = add_event_features(df)

    # Step 2: Add weather-related features
    df = add_weather_features(df)

    # Step 3: Add enhanced temporal features
    df = add_temporal_features(df)

    # Select columns to keep
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
        "has_event",
        "is_raining",
        "is_raining_downtown",
        "is_evening_rush",
        "is_weekend_hour",
        "is_rush_hour_downtown",
        "is_downtown_hour",
    ]
    df = df[columns_to_keep]

    # Save the enhanced data
    output_file = (
        DATA_PROCESSED_DIR
        / f"enhanced_yellow_tripdata_2019-{month_name.lower()}.parquet"
    )
    df.to_parquet(output_file, engine="pyarrow", compression="snappy")
    logger.info(f"Saved enhanced {month_name} data to {output_file}")


def main():
    # Process each month's data
    months = ["January", "February", "March"]
    for month in months:
        process_month(month)


if __name__ == "__main__":
    main()
