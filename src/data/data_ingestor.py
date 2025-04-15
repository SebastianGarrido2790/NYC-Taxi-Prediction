import dask.dataframe as dd
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim"
DATA_EXTERNAL_DIR = BASE_DIR / "data" / "external"

# Ensure interim directory exists
DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def load_taxi_zone_lookup():
    """Load the Taxi Zone Lookup Table and return Manhattan LocationIDs."""
    lookup_file = DATA_EXTERNAL_DIR / "taxi_zone_lookup.csv"
    if not lookup_file.exists():
        raise FileNotFoundError(f"Taxi Zone Lookup file not found at {lookup_file}")

    lookup_df = pd.read_csv(lookup_file)
    manhattan_zones = lookup_df[lookup_df["Borough"] == "Manhattan"][
        "LocationID"
    ].tolist()
    logger.info(f"Found {len(manhattan_zones)} Manhattan zones: {manhattan_zones}")
    return set(manhattan_zones)


def validate_and_clean_data(df, month, manhattan_zones):
    """Validate and clean the Dask DataFrame."""
    logger.info(f"Starting validation and cleaning for {month} data...")

    # Log initial row count and missing values
    initial_rows = len(df)
    missing_stats = df.isna().mean().compute() * 100
    logger.info(f"Missing value percentages:\n{missing_stats}")

    # Drop columns with 100% missing values
    columns_to_drop = missing_stats[missing_stats == 100].index
    if len(columns_to_drop) > 0:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Dropped columns with 100% missing values: {columns_to_drop}")

    # 1. Filter for correct time range (2019-01 to 2019-03)
    start_date = pd.to_datetime("2019-01-01")
    end_date = pd.to_datetime("2019-03-31 23:59:59")
    df = df[
        (df["tpep_pickup_datetime"] >= start_date)
        & (df["tpep_pickup_datetime"] <= end_date)
    ]
    logger.info(f"After time range filter: {len(df)} rows remaining.")

    # 2. Filter for Manhattan zones (pickup or dropoff in Manhattan)
    df = df[
        (df["PULocationID"].isin(manhattan_zones))
        | (df["DOLocationID"].isin(manhattan_zones))
    ]
    logger.info(f"After Manhattan zone filter: {len(df)} rows remaining.")

    # 3. Validate timestamp consistency
    df = df[df["tpep_dropoff_datetime"] >= df["tpep_pickup_datetime"]]
    logger.info(f"After timestamp consistency check: {len(df)} rows remaining.")

    # 4. Handle missing values
    # Drop rows where critical columns are missing
    critical_columns = ["tpep_pickup_datetime", "PULocationID", "DOLocationID"]
    df = df.dropna(subset=critical_columns)
    logger.info(f"After dropping missing critical columns: {len(df)} rows remaining.")

    # Impute missing values for other columns
    try:
        # Fill missing passenger_count with median (computed after filtering)
        passenger_median = df["passenger_count"].median_approximate().compute()
        logger.info(
            f"Imputing passenger_count with approximate median: {passenger_median}"
        )
        df["passenger_count"] = df["passenger_count"].fillna(passenger_median)

        trip_distance_median = df["trip_distance"].median_approximate().compute()
        logger.info(
            f"Imputing trip_distance with approximate median: {trip_distance_median}"
        )
        df["trip_distance"] = df["trip_distance"].fillna(trip_distance_median)

        fare_amount_median = df["fare_amount"].median_approximate().compute()
        logger.info(
            f"Imputing fare_amount with approximate median: {fare_amount_median}"
        )
        df["fare_amount"] = df["fare_amount"].fillna(fare_amount_median)

        total_amount_median = df["total_amount"].median_approximate().compute()
        logger.info(
            f"Imputing total_amount with approximate median: {total_amount_median}"
        )
        df["total_amount"] = df["total_amount"].fillna(total_amount_median)

        # Fill congestion_surcharge with 0 (introduced in 2019, so many trips may not have it)
        df["congestion_surcharge"] = df["congestion_surcharge"].fillna(0)

    except Exception as e:
        logger.error(f"Failed to impute missing values: {e}")
        raise

    # 5. Correct data types
    # Downcast numeric columns to save memory
    df["VendorID"] = df["VendorID"].astype("int32")
    df["PULocationID"] = df["PULocationID"].astype("int32")
    df["DOLocationID"] = df["DOLocationID"].astype("int32")
    df["passenger_count"] = df["passenger_count"].astype("float32")
    df["trip_distance"] = df["trip_distance"].astype("float32")
    df["fare_amount"] = df["fare_amount"].astype("float32")
    df["total_amount"] = df["total_amount"].astype("float32")
    df["congestion_surcharge"] = df["congestion_surcharge"].astype("float32")
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("category")

    # 6. Validate categorical columns
    df = df[df["VendorID"].isin([1, 2])]
    df = df[df["RatecodeID"].isin([1, 2, 3, 4, 5, 6])]
    df = df[df["payment_type"].isin([1, 2, 3, 4, 5, 6])]
    logger.info(f"After categorical validation: {len(df)} rows remaining.")

    # 7. Remove outliers
    # Trip distance: Remove negative or unreasonably large values (e.g., > 100 miles in Manhattan)
    df = df[(df["trip_distance"] >= 0) & (df["trip_distance"] <= 100)]
    # Passenger count: Remove values < 0 or > 6 (typical taxi capacity)
    df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 6)]
    # Fare amount: Remove negative or unreasonably large values (e.g., > $1000)
    df = df[(df["fare_amount"] >= 0) & (df["fare_amount"] <= 1000)]
    df = df[(df["total_amount"] >= 0) & (df["total_amount"] <= 1000)]
    df = df[df["tip_amount"] >= 0]
    df = df[df["tolls_amount"] >= 0]
    logger.info(f"After outlier removal: {len(df)} rows remaining.")

    # 8. Compute the number of rows after cleaning
    num_rows = len(df)
    logger.info(
        f"Cleaned {month} data: {num_rows} rows remaining ({(num_rows/initial_rows)*100:.2f}% of original."
    )

    return df


def process_month(month_file, month_name, manhattan_zones):
    """Process a single month's data and save the cleaned version."""
    logger.info(f"Processing {month_name} data from {month_file}...")

    # Load the Parquet file into a Dask DataFrame with dynamic partitions
    df = dd.read_parquet(month_file)
    npartitions = max(1, len(df) // 1_000_000)  # 1 partition per million rows
    df = df.repartition(npartitions=npartitions)
    logger.info(f"Repartitioned DataFrame to {npartitions} partitions.")

    # Validate and clean the data
    df_cleaned = validate_and_clean_data(df, month_name, manhattan_zones)

    # Save the cleaned data to interim directory
    output_file = (
        DATA_INTERIM_DIR / f"cleaned_yellow_tripdata_2019-{month_name.lower()}.parquet"
    )
    df_cleaned.to_parquet(output_file, engine="pyarrow", compression="snappy")
    logger.info(f"Saved cleaned {month_name} data to {output_file}")


def main():
    # Load Manhattan zones
    manhattan_zones = load_taxi_zone_lookup()

    # Process each month's data
    months = [
        ("yellow_tripdata_2019-01.parquet", "January"),
        ("yellow_tripdata_2019-02.parquet", "February"),
        ("yellow_tripdata_2019-03.parquet", "March"),
    ]

    for month_file, month_name in months:
        month_path = DATA_RAW_DIR / month_file
        if not month_path.exists():
            logger.error(f"File {month_path} not found. Skipping...")
            continue
        process_month(month_path, month_name, manhattan_zones)


if __name__ == "__main__":
    main()
