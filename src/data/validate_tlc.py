from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yellow_tripdata_2025-01_cleaned.parquet"
)

BATCH_SIZE = 100_000


def main():
    if not FILE.exists():
        raise FileNotFoundError(FILE)

    parquet = pq.ParquetFile(FILE)

    total_rows = 0
    outside_january = 0
    invalid_duration = 0
    invalid_distance = 0
    invalid_fare = 0
    inconsistent_duration = 0

    required_columns = {
        "trip_duration_seconds",
        "trip_duration_minutes",
        "is_valid_duration",
        "is_valid_distance",
        "is_valid_fare",
        "pickup_in_january",
        "cleaned_trip_distance",
        "cleaned_fare_amount",
        "cleaned_passenger_count",
    }

    missing_columns = required_columns - set(parquet.schema.names)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        total_rows += len(df)

        pickup = df["tpep_pickup_datetime"]

        outside_january += int(
            (
                pickup.lt(pd.Timestamp("2025-01-01"))
                | pickup.ge(pd.Timestamp("2025-02-01"))
            ).sum()
        )

        invalid_duration += int((~df["is_valid_duration"]).sum())
        invalid_distance += int((~df["is_valid_distance"]).sum())
        invalid_fare += int((~df["is_valid_fare"]).sum())

        expected_duration = (
            df["trip_duration_seconds"].gt(0)
            & df["trip_duration_seconds"].le(86400)
        ).fillna(False)

        inconsistent_duration += int(
            (df["is_valid_duration"] != expected_duration).sum()
        )

    print("\n" + "=" * 55)
    print("MOVEIQ DATASET VALIDATION")
    print("=" * 55)

    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(parquet.schema.names)}")
    print(f"Rows outside January: {outside_january:,}")
    print(f"Invalid duration: {invalid_duration:,}")
    print(f"Invalid distance: {invalid_distance:,}")
    print(f"Invalid fare: {invalid_fare:,}")
    print(f"Inconsistent duration flags: {inconsistent_duration:,}")

    checks = {
        "Expected row count": total_rows == 3_475_204,
        "No pickups outside January": outside_january == 0,
        "Required columns exist": not missing_columns,
        "Duration flags consistent": inconsistent_duration == 0,
    }

    print("\nValidation checks:")

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} - {name}")

    if all(checks.values()):
        print("\nDataset validation successful.")
    else:
        print("\nDataset validation needs attention.")


if __name__ == "__main__":
    main()
