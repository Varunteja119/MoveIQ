import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/raw/yellow_tripdata_2025-01.parquet"
OUTPUT_FILE = ROOT / "data/processed/jan_2025_audit.json"

BATCH_SIZE = 100_000
JAN_START = pd.Timestamp("2025-01-01")
FEB_START = pd.Timestamp("2025-02-01")

COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]


def get_records(df, metric, largest=True, n=10):
    if df.empty:
        return []

    result = (
        df.nlargest(n, metric)
        if largest
        else df.nsmallest(n, metric)
    )

    return result.to_dict(orient="records")


def make_json_safe(records):
    safe_records = []

    for record in records:
        safe = {}

        for key, value in record.items():
            if pd.isna(value):
                safe[key] = None
            elif isinstance(value, pd.Timestamp):
                safe[key] = str(value)
            elif hasattr(value, "item"):
                safe[key] = value.item()
            else:
                safe[key] = value

        safe_records.append(safe)

    return safe_records


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    parquet = pq.ParquetFile(INPUT_FILE)

    outside_january = 0
    before_january = 0
    after_january = 0

    vendor_stats = {}

    top = {
        "highest_fares": [],
        "lowest_fares": [],
        "longest_distances": [],
        "shortest_distances": [],
        "longest_durations": [],
        "negative_durations": [],
    }

    for batch_number, batch in enumerate(
        parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=COLUMNS,
        ),
        start=1,
    ):
        df = batch.to_pandas()

        pickup = df["tpep_pickup_datetime"]

        before = pickup < JAN_START
        after = pickup >= FEB_START

        before_january += int(before.sum())
        after_january += int(after.sum())
        outside_january += int((before | after).sum())

        # Missing-value distribution by vendor
        for vendor, group in df.groupby("VendorID", dropna=False):
            key = str(vendor)

            if key not in vendor_stats:
                vendor_stats[key] = {
                    "records": 0,
                    "missing_passenger_count": 0,
                    "missing_ratecode": 0,
                    "missing_store_and_fwd_flag": 0,
                    "missing_congestion_surcharge": 0,
                    "missing_airport_fee": 0,
                }

            stats = vendor_stats[key]

            stats["records"] += len(group)

            stats["missing_passenger_count"] += int(
                group["passenger_count"].isna().sum()
            )

            stats["missing_ratecode"] += int(
                group["RatecodeID"].isna().sum()
            )

            stats["missing_store_and_fwd_flag"] += int(
                group["store_and_fwd_flag"].isna().sum()
            )

            stats["missing_congestion_surcharge"] += int(
                group["congestion_surcharge"].isna().sum()
            )

            stats["missing_airport_fee"] += int(
                group["Airport_fee"].isna().sum()
            )

        df["duration_seconds"] = (
            df["tpep_dropoff_datetime"]
            - df["tpep_pickup_datetime"]
        ).dt.total_seconds()

        # Congestion fee anomalies
        # These categories help reconcile the fee counts.
        if "cbd_fee_negative" not in top:
            top["cbd_fee_negative"] = []

        negative_fees = df[df["cbd_congestion_fee"] < 0]

        top["cbd_fee_negative"].extend(
            get_records(negative_fees, "cbd_congestion_fee",
                        largest=False)
        )

        # Keep only the most extreme records per metric.
        checks = {
            "highest_fares": ("fare_amount", True),
            "lowest_fares": ("fare_amount", False),
            "longest_distances": ("trip_distance", True),
            "shortest_distances": ("trip_distance", False),
            "longest_durations": ("duration_seconds", True),
            "negative_durations": ("duration_seconds", False),
        }

        for name, (metric, largest) in checks.items():
            candidates = get_records(df, metric, largest)

            top[name].extend(candidates)

            combined = pd.DataFrame(top[name])

            top[name] = get_records(
                combined,
                metric,
                largest,
            )

        print(
            f"Audited batch {batch_number} | "
            f"{batch_number * BATCH_SIZE:,} rows approximately"
        )

        del df

    # Final audit report
    report = {
        "dataset": INPUT_FILE.name,
        "outside_january": {
            "total": outside_january,
            "before_january": before_january,
            "on_or_after_february": after_january,
        },
        "vendor_missing_value_analysis": vendor_stats,
        "extreme_records": {
            name: make_json_safe(records)
            for name, records in top.items()
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nAudit completed.")
    print(f"Report saved to: {OUTPUT_FILE}")

    print("\nRecords outside January:", outside_january)
    print("Records before January:", before_january)
    print("Records on/after February:", after_january)

    print("\nVendor missing-value analysis:")

    print(json.dumps(vendor_stats, indent=4))


if __name__ == "__main__":
    main()
