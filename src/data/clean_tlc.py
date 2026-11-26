from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2025-01.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "yellow_tripdata_2025-01_cleaned.parquet"
REPORT_FILE = OUTPUT_DIR / "cleaning_report.json"

BATCH_SIZE = 100_000

# Practical limits for this first analytical dataset
MAX_TRIP_DURATION_SECONDS = 24 * 60 * 60
MAX_TRIP_DISTANCE_MILES = 500
MAX_FARE_AMOUNT = 1_000

JANUARY_START = pd.Timestamp("2025-01-01")
FEBRUARY_START = pd.Timestamp("2025-02-01")


# --------------------------------------------------
# CLEANING FUNCTION
# --------------------------------------------------

def clean_batch(batch: pd.DataFrame):
    """
    Preserve original columns and add cleaned analytical
    columns and validity flags.
    """
    # Keep integer columns consistent across batches,
    # including batches containing missing values.
    batch["passenger_count"] = batch["passenger_count"].astype("Int64")
    batch["RatecodeID"] = batch["RatecodeID"].astype("Int64")

    # Calculate trip duration
    duration_seconds = (
        batch["tpep_dropoff_datetime"]
        - batch["tpep_pickup_datetime"]
    ).dt.total_seconds()

    # Identify January pickup records
    pickup_in_january = (
        batch["tpep_pickup_datetime"].ge(JANUARY_START)
        & batch["tpep_pickup_datetime"].lt(FEBRUARY_START)
    ).fillna(False)

    # Validate duration: positive and no longer than 24 hours
    valid_duration = (
        duration_seconds.gt(0)
        & duration_seconds.le(MAX_TRIP_DURATION_SECONDS)
    ).fillna(False)

    # Validate distance: zero is retained, negative and
    # extremely large values are excluded from distance analysis
    valid_distance = (
        batch["trip_distance"].ge(0)
        & batch["trip_distance"].le(MAX_TRIP_DISTANCE_MILES)
    ).fillna(False)

    # Validate fare: negative and extremely high fares are
    # excluded from normal fare analysis
    valid_fare = (
        batch["fare_amount"].ge(0)
        & batch["fare_amount"].le(MAX_FARE_AMOUNT)
    ).fillna(False)

    # Passenger count: zero or negative values become unknown
    valid_passenger_count = batch["passenger_count"].gt(0).fillna(False)

    # Add analytical columns
    batch["trip_duration_seconds"] = duration_seconds
    batch["trip_duration_minutes"] = duration_seconds / 60

    batch["is_valid_duration"] = valid_duration
    batch["is_valid_distance"] = valid_distance
    batch["is_valid_fare"] = valid_fare
    batch["pickup_in_january"] = pickup_in_january

    # Keep original fields unchanged.
    # Invalid values become null only in the cleaned columns.
    batch["cleaned_trip_distance"] = batch["trip_distance"].where(
        valid_distance
    )

    batch["cleaned_fare_amount"] = batch["fare_amount"].where(
        valid_fare
    )

    batch["cleaned_passenger_count"] = batch["passenger_count"].where(
        valid_passenger_count
    )

    # Keep only trips whose pickup occurred in January 2025
    batch = batch.loc[pickup_in_january].copy()

    return batch


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}\n"
            "Check the filename and data/raw directory."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(INPUT_FILE)

    writer = None

    total_input_rows = 0
    total_output_rows = 0

    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "batch_size": BATCH_SIZE,
        "total_input_rows": 0,
        "total_output_rows": 0,
        "rows_excluded_outside_january": 0,
        "invalid_duration": 0,
        "invalid_distance": 0,
        "invalid_fare": 0,
        "invalid_passenger_count": 0,
        "missing_passenger_count": 0,
        "missing_ratecode": 0,
        "missing_store_and_fwd_flag": 0,
        "missing_congestion_surcharge": 0,
        "missing_airport_fee": 0,
    }

    print("\nStarting MoveIQ data-cleaning pipeline...")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Batch size: {BATCH_SIZE:,}\n")

    try:
        for batch_number, arrow_batch in enumerate(
            parquet_file.iter_batches(batch_size=BATCH_SIZE),
            start=1,
        ):

            batch = arrow_batch.to_pandas()

            input_count = len(batch)
            total_input_rows += input_count

            # Count quality issues before filtering
            report["invalid_duration"] += int(
                (
                    (batch["tpep_dropoff_datetime"]
                     - batch["tpep_pickup_datetime"])
                    .dt.total_seconds()
                    .le(0)
                    |
                    (
                        (batch["tpep_dropoff_datetime"]
                         - batch["tpep_pickup_datetime"])
                        .dt.total_seconds()
                        > MAX_TRIP_DURATION_SECONDS
                    )
                ).fillna(False).sum()
            )

            report["invalid_distance"] += int(
                (
                    batch["trip_distance"].lt(0)
                    | batch["trip_distance"].gt(MAX_TRIP_DISTANCE_MILES)
                ).fillna(False).sum()
            )

            report["invalid_fare"] += int(
                (
                    batch["fare_amount"].lt(0)
                    | batch["fare_amount"].gt(MAX_FARE_AMOUNT)
                ).fillna(False).sum()
            )

            report["invalid_passenger_count"] += int(
                batch["passenger_count"].le(0).fillna(False).sum()
            )

            report["missing_passenger_count"] += int(
                batch["passenger_count"].isna().sum()
            )

            report["missing_ratecode"] += int(
                batch["RatecodeID"].isna().sum()
            )

            report["missing_store_and_fwd_flag"] += int(
                batch["store_and_fwd_flag"].isna().sum()
            )

            report["missing_congestion_surcharge"] += int(
                batch["congestion_surcharge"].isna().sum()
            )

            report["missing_airport_fee"] += int(
                batch["Airport_fee"].isna().sum()
            )

            # Apply cleaning rules
            cleaned_batch = clean_batch(batch)

            output_count = len(cleaned_batch)

            report["rows_excluded_outside_january"] += (
                input_count - output_count
            )

            total_output_rows += output_count

            if output_count > 0:

                table = pa.Table.from_pandas(
                    cleaned_batch,
                    preserve_index=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        OUTPUT_FILE,
                        table.schema,
                        compression="snappy",
                    )

                writer.write_table(table)

            print(
                f"Batch {batch_number}: "
                f"Read {input_count:,} | "
                f"January records retained {output_count:,}"
            )

    finally:
        if writer is not None:
            writer.close()

    if total_output_rows == 0:
        raise RuntimeError(
            "No January records were written. Check the pickup dates."
        )

    # Final report
    report["total_input_rows"] = total_input_rows
    report["total_output_rows"] = total_output_rows
    report["cleaning_rules"] = {
        "pickup_date": "January 2025 only",
        "duration": "Greater than 0 and at most 24 hours",
        "distance": "Between 0 and 500 miles",
        "fare": "Between 0 and 1000",
        "passenger_count": "Must be greater than 0; otherwise cleaned value is null",
        "missing_values": "Preserved as missing; no arbitrary imputation",
        "original_columns": "Preserved without overwriting",
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print("\n" + "=" * 55)
    print("MOVEIQ CLEANING COMPLETED")
    print("=" * 55)

    print(f"Original rows: {total_input_rows:,}")
    print(f"January rows retained: {total_output_rows:,}")
    print(
        "Rows outside January excluded: "
        f"{report['rows_excluded_outside_january']:,}"
    )

    print(f"\nCleaned dataset: {OUTPUT_FILE}")
    print(f"Cleaning report: {REPORT_FILE}")


if __name__ == "__main__":
    main()