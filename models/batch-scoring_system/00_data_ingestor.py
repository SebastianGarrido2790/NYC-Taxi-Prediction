import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim" / "production"
DATA_EXTERNAL_DIR = BASE_DIR / "data" / "external"

# Ensure interim directory exists
DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Set up MLflow experiment
EXPERIMENT_NAME = "NYC_Taxi_Demand_Production"
mlflow.set_experiment(EXPERIMENT_NAME)


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


def validate_and_clean_data(df, current_time, time_window_hours, manhattan_zones):
    """Validate and clean the Dask DataFrame for the specified time window."""
    logger.info(f"Starting validation and cleaning for data up to {current_time}...")

    # Log initial row count and missing values
    initial_rows = len(df)
    missing_stats = df.isna().mean().compute() * 100
    logger.info(f"Missing value percentages:\n{missing_stats}")
    mlflow.log_metric("initial_rows", initial_rows)

    # Drop columns with 100% missing values
    columns_to_drop = missing_stats[missing_stats == 100].index
    if len(columns_to_drop) > 0:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Dropped columns with 100% missing values: {columns_to_drop}")

    # 1. Filter for the specified time window (e.g., last 24 hours)
    start_time = current_time - timedelta(hours=time_window_hours)
    df = df[
        (df["tpep_pickup_datetime"] >= start_time)
        & (df["tpep_pickup_datetime"] <= current_time)
    ]
    logger.info(
        f"After time window filter ({start_time} to {current_time}): {len(df)} rows remaining."
    )

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
        f"Cleaned data up to {current_time}: {num_rows} rows remaining ({(num_rows/initial_rows)*100:.2f}% of original)."
    )
    mlflow.log_metric("cleaned_rows", num_rows)

    return df


def process_data(
    input_file, current_time, time_window_hours, manhattan_zones, output_filename
):
    """Process data for the specified time window and save to interim directory."""
    logger.info(f"Processing data from {input_file} up to {current_time}...")

    # Load the Parquet file into a Dask DataFrame with dynamic partitions
    df = dd.read_parquet(input_file)
    npartitions = max(1, len(df) // 1_000_000)  # 1 partition per million rows
    df = df.repartition(npartitions=npartitions)
    logger.info(f"Repartitioned DataFrame to {npartitions} partitions.")

    # Validate and clean the data for the specified time window
    df_cleaned = validate_and_clean_data(
        df, current_time, time_window_hours, manhattan_zones
    )

    # Save the cleaned data to interim/production directory
    output_file = DATA_INTERIM_DIR / output_filename
    df_cleaned.to_parquet(output_file, engine="pyarrow", compression="snappy")
    logger.info(f"Saved cleaned data to {output_file}")
    mlflow.log_artifact(output_file)

    return output_file


def backfill_historical_data(manhattan_zones):
    """Backfill historical data from January to March 2019."""
    logger.info("Starting historical data backfill...")

    # Define historical months to process
    months = ["01", "02", "03"]
    for month in months:
        input_file = DATA_RAW_DIR / f"yellow_tripdata_2019-{month}.parquet"
        if not input_file.exists():
            logger.warning(f"Historical file {input_file} not found. Skipping...")
            continue

        # Define the time range for the month
        start_time = pd.to_datetime(f"2019-{month}-01 00:00:00")
        end_time = start_time + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
        time_window_hours = (end_time - start_time).total_seconds() / 3600  # Full month

        logger.info(
            f"Backfilling data for 2019-{month} (from {start_time} to {end_time})..."
        )
        output_filename = f"cleaned_yellow_tripdata_2019-{month}.parquet"
        process_data(
            input_file, end_time, time_window_hours, manhattan_zones, output_filename
        )


def main():
    with mlflow.start_run(run_name="Data_Ingestion"):
        # Load Manhattan zones
        manhattan_zones = load_taxi_zone_lookup()

        # Step 1: Backfill historical data if not already done
        historical_files = list(
            DATA_INTERIM_DIR.glob("cleaned_yellow_tripdata_2019-*.parquet")
        )
        historical_months = {
            f.name.split("cleaned_yellow_tripdata_2019-")[1].split(".parquet")[0]
            for f in historical_files
        }
        if not all(month in historical_months for month in ["01", "02", "03"]):
            backfill_historical_data(manhattan_zones)

        # Step 2: Process the latest data
        # Simulate the current time in production (set to 2019-04-01 00:00:00)
        current_time = pd.to_datetime("2019-04-01 00:00:00")
        time_window_hours = 1  # Process the last hour of data

        # Load the March 2019 raw data for the latest hour
        month_file = DATA_RAW_DIR / "yellow_tripdata_2019-03.parquet"
        if not month_file.exists():
            logger.error(f"File {month_file} not found. Exiting...")
            raise FileNotFoundError(f"File {month_file} not found.")

        # Process the latest data
        timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"cleaned_yellow_tripdata_{timestamp_str}.parquet"
        output_file = process_data(
            month_file,
            current_time,
            time_window_hours,
            manhattan_zones,
            output_filename,
        )

    logger.info("Data ingestion pipeline completed.")


if __name__ == "__main__":
    main()
