import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2025-01.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jan_2025_profile.json"
)

BATCH_SIZE = 100_000


def profile_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    parquet_file = pq.ParquetFile(INPUT_FILE)

    total_rows = parquet_file.metadata.num_rows

    null_counts = {
        column: 0
        for column in parquet_file.schema_arrow.names
    }

    metrics = {
        "passenger_count_zero": 0,
        "passenger_count_negative": 0,
        "trip_distance_zero_or_negative": 0,
        "fare_amount_negative": 0,
        "duration_zero": 0,
        "duration_negative": 0,
        "same_pickup_dropoff_zone": 0,
        "congestion_fee_zero": 0,
        "congestion_fee_positive": 0,
        "duration_valid": 0,
    }

    statistics = {
        "duration_seconds": {
            "min": None,
            "max": None,
            "sum": 0.0,
        },
        "trip_distance": {
            "min": None,
            "max": None,
            "sum": 0.0,
        },
        "fare_amount": {
            "min": None,
            "max": None,
            "sum": 0.0,
        },
    }

    pickup_min = None
    pickup_max = None

    processed_rows = 0

    print(f"Total records: {total_rows:,}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print("Starting profiling...\n")

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ),
        start=1,
    ):

        df = batch.to_pandas()

        processed_rows += len(df)

        # Missing-value counts
        missing = df.isna().sum()

        for column, count in missing.items():
            null_counts[column] += int(count)

        # Trip duration in seconds
        duration = (
            df["tpep_dropoff_datetime"]
            - df["tpep_pickup_datetime"]
        ).dt.total_seconds()

        valid_duration = duration.dropna()

        metrics["duration_zero"] += int(
            (valid_duration == 0).sum()
        )

        metrics["duration_negative"] += int(
            (valid_duration < 0).sum()
        )

        positive_duration = valid_duration[
            valid_duration > 0
        ]

        metrics["duration_valid"] += len(
            positive_duration
        )

        if not positive_duration.empty:
            stat = statistics["duration_seconds"]

            stat["min"] = (
                float(positive_duration.min())
                if stat["min"] is None
                else min(
                    stat["min"],
                    float(positive_duration.min()),
                )
            )

            stat["max"] = (
                float(positive_duration.max())
                if stat["max"] is None
                else max(
                    stat["max"],
                    float(positive_duration.max()),
                )
            )

            stat["sum"] += float(
                positive_duration.sum()
            )

        # Passenger count
        passengers = df["passenger_count"]

        metrics["passenger_count_zero"] += int(
            (passengers == 0).sum()
        )

        metrics["passenger_count_negative"] += int(
            (passengers < 0).sum()
        )

        # Trip distance
        distance = df["trip_distance"].dropna()

        metrics["trip_distance_zero_or_negative"] += int(
            (df["trip_distance"] <= 0).sum()
        )

        if not distance.empty:
            stat = statistics["trip_distance"]

            stat["min"] = (
                float(distance.min())
                if stat["min"] is None
                else min(stat["min"], float(distance.min()))
            )

            stat["max"] = (
                float(distance.max())
                if stat["max"] is None
                else max(stat["max"], float(distance.max()))
            )

            stat["sum"] += float(distance.sum())

        # Fare amount
        fare = df["fare_amount"].dropna()

        metrics["fare_amount_negative"] += int(
            (df["fare_amount"] < 0).sum()
        )

        if not fare.empty:
            stat = statistics["fare_amount"]

            stat["min"] = (
                float(fare.min())
                if stat["min"] is None
                else min(stat["min"], float(fare.min()))
            )

            stat["max"] = (
                float(fare.max())
                if stat["max"] is None
                else max(stat["max"], float(fare.max()))
            )

            stat["sum"] += float(fare.sum())

        # Pickup and drop-off locations
        same_zone = (
            df["PULocationID"].notna()
            & df["DOLocationID"].notna()
            & (
                df["PULocationID"]
                == df["DOLocationID"]
            )
        )

        metrics["same_pickup_dropoff_zone"] += int(
            same_zone.sum()
        )

        # Congestion fee
        fee = df["cbd_congestion_fee"]

        metrics["congestion_fee_zero"] += int(
            (fee == 0).sum()
        )

        metrics["congestion_fee_positive"] += int(
            (fee > 0).sum()
        )

        # Pickup date range
        pickup = df["tpep_pickup_datetime"].dropna()

        if not pickup.empty:
            current_min = pickup.min()
            current_max = pickup.max()

            pickup_min = (
                current_min
                if pickup_min is None
                else min(pickup_min, current_min)
            )

            pickup_max = (
                current_max
                if pickup_max is None
                else max(pickup_max, current_max)
            )

        del df

        print(
            f"Processed batch {batch_number} | "
            f"{processed_rows:,}/{total_rows:,} rows"
        )

    # Calculate averages
    for name, stat in statistics.items():
        if name == "duration_seconds":
            count = metrics["duration_valid"]
        elif name == "trip_distance":
            count = total_rows - null_counts["trip_distance"]
            # This average includes zero/negative values.
        else:
            count = total_rows - null_counts["fare_amount"]

        stat["count"] = count

        stat["mean"] = (
            stat["sum"] / count
            if count > 0
            else None
        )

    report = {
        "dataset": INPUT_FILE.name,
        "total_rows": total_rows,
        "processed_rows": processed_rows,
        "batch_size": BATCH_SIZE,
        "pickup_date_range": {
            "min": str(pickup_min),
            "max": str(pickup_max),
        },
        "missing_values": null_counts,
        "quality_metrics": metrics,
        "statistics": statistics,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nProfiling completed.")
    print(f"Report saved to: {OUTPUT_FILE}")

    print("\nQuality metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:,}")


if __name__ == "__main__":
    profile_dataset()
