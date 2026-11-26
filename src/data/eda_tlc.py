from pathlib import Path
from collections import Counter
import json

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yellow_tripdata_2025-01_cleaned.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "eda"

BATCH_SIZE = 100_000


def update_stats(stats, series):
    values = series.dropna()

    if values.empty:
        return

    stats["count"] += len(values)
    stats["sum"] += float(values.sum())
    stats["min"] = min(stats["min"], float(values.min()))
    stats["max"] = max(stats["max"], float(values.max()))


def finalize_stats(stats):
    if stats["count"] == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }

    return {
        "count": stats["count"],
        "mean": round(stats["sum"] / stats["count"], 2),
        "min": round(stats["min"], 2),
        "max": round(stats["max"], 2),
    }


def save_counter(counter, column_name, output_file):
    df = pd.DataFrame(
        counter.items(),
        columns=[column_name, "trip_count"],
    )

    df = df.sort_values("trip_count", ascending=False)
    df.to_csv(output_file, index=False)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(INPUT_FILE)

    hourly_demand = Counter()
    weekday_demand = Counter()
    pickup_zones = Counter()
    dropoff_zones = Counter()
    payment_types = Counter()

    fare_stats = {
        "count": 0,
        "sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }

    distance_stats = {
        "count": 0,
        "sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }

    duration_stats = {
        "count": 0,
        "sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }

    total_rows = 0

    print("\nStarting MoveIQ exploratory data analysis...\n")

    for batch_number, batch in enumerate(
        parquet.iter_batches(batch_size=BATCH_SIZE),
        start=1,
    ):
        df = batch.to_pandas()
        total_rows += len(df)

        pickup = df["tpep_pickup_datetime"]

        # Demand by pickup hour
        hourly_demand.update(
            pickup.dt.hour.value_counts().to_dict()
        )

        # Demand by weekday
        weekday_demand.update(
            pickup.dt.dayofweek.value_counts().to_dict()
        )

        # Most popular pickup and drop-off zones
        pickup_zones.update(
            df["PULocationID"].value_counts().to_dict()
        )

        dropoff_zones.update(
            df["DOLocationID"].value_counts().to_dict()
        )

        # Payment methods
        payment_types.update(
            df["payment_type"].value_counts().to_dict()
        )

        # Independent validity filters for each metric
        update_stats(
            fare_stats,
            df.loc[df["is_valid_fare"], "cleaned_fare_amount"],
        )

        update_stats(
            distance_stats,
            df.loc[df["is_valid_distance"], "cleaned_trip_distance"],
        )

        update_stats(
            duration_stats,
            df.loc[df["is_valid_duration"], "trip_duration_minutes"],
        )

        print(
            f"Processed batch {batch_number} | "
            f"Total rows: {total_rows:,}"
        )

    # Save aggregated CSV files
    save_counter(
        hourly_demand,
        "pickup_hour",
        OUTPUT_DIR / "hourly_demand.csv",
    )

    weekday_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    weekday_counter = Counter({
        weekday_names[int(day)]: count
        for day, count in weekday_demand.items()
    })

    save_counter(
        weekday_counter,
        "weekday",
        OUTPUT_DIR / "weekday_demand.csv",
    )

    save_counter(
        pickup_zones,
        "pickup_location_id",
        OUTPUT_DIR / "pickup_zones.csv",
    )

    save_counter(
        dropoff_zones,
        "dropoff_location_id",
        OUTPUT_DIR / "dropoff_zones.csv",
    )

    payment_labels = {
        1: "Credit card",
        2: "Cash",
        3: "No charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided trip",
    }

    payment_counter = Counter({
        payment_labels.get(int(code), f"Other ({code})"): count
        for code, count in payment_types.items()
    })

    save_counter(
        payment_counter,
        "payment_type",
        OUTPUT_DIR / "payment_types.csv",
    )

    # Create summary report
    summary = {
        "dataset": "NYC TLC Yellow Taxi January 2025",
        "total_rows_analyzed": total_rows,
        "trip_metrics": {
            "fare_amount": finalize_stats(fare_stats),
            "trip_distance_miles": finalize_stats(distance_stats),
            "trip_duration_minutes": finalize_stats(duration_stats),
        },
        "busiest_hour": max(
            hourly_demand,
            key=hourly_demand.get,
        ),
        "busiest_weekday": max(
            weekday_demand,
            key=weekday_demand.get,
        ),
        "top_5_pickup_zones": pickup_zones.most_common(5),
        "top_5_dropoff_zones": dropoff_zones.most_common(5),
        "payment_type_distribution": dict(payment_counter),
    }

    with open(
        OUTPUT_DIR / "eda_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("MOVEIQ EDA COMPLETED")
    print("=" * 55)

    print(f"Rows analyzed: {total_rows:,}")

    print("\nTrip metrics:")
    print(json.dumps(summary["trip_metrics"], indent=4))

    print(f"\nBusiest pickup hour: {summary['busiest_hour']}:00")
    print(f"Busiest weekday: {summary['busiest_weekday']}")

    print("\nTop 5 pickup zones:")
    for zone, count in pickup_zones.most_common(5):
        print(f"Zone {zone}: {count:,} trips")

    print("\nTop 5 drop-off zones:")
    for zone, count in dropoff_zones.most_common(5):
        print(f"Zone {zone}: {count:,} trips")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
