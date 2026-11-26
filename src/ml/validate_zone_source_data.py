from pathlib import Path
from datetime import datetime
import re

import duckdb

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "ml" / "raw"
LOOKUP_FILE = ROOT / "data" / "reference" / "taxi_zone_lookup.csv"

con = duckdb.connect()

lookup_path = LOOKUP_FILE.as_posix()

files = sorted(DATA_DIR.glob("yellow_tripdata_*.parquet"))

results = []

for file_path in files:

    match = re.search(r"yellow_tripdata_(\d{4})-(\d{2})", file_path.name)

    if not match:
        continue

    year, month = map(int, match.groups())

    month_start = datetime(year, month, 1)

    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)

    parquet_path = file_path.as_posix()

    result = con.execute(f"""
        SELECT
            COUNT(*) AS total_records,
            MIN(tpep_pickup_datetime) AS earliest_pickup,
            MAX(tpep_pickup_datetime) AS latest_pickup,

            COUNT(*) FILTER (
                WHERE tpep_pickup_datetime < TIMESTAMP '{month_start}'
                   OR tpep_pickup_datetime >= TIMESTAMP '{next_month}'
            ) AS outside_filename_month,

            COUNT(*) FILTER (
                WHERE tpep_pickup_datetime >= TIMESTAMP '{month_start}'
                  AND tpep_pickup_datetime < TIMESTAMP '{next_month}'
            ) AS records_inside_filename_month,

            COUNT(*) FILTER (
                WHERE PULocationID IS NULL
            ) AS missing_pickup_zone,

            COUNT(*) FILTER (
                WHERE PULocationID IS NOT NULL
                  AND z.LocationID IS NULL
            ) AS unknown_pickup_zone

        FROM read_parquet('{parquet_path}') t

        LEFT JOIN read_csv_auto('{lookup_path}') z
            ON t.PULocationID = z.LocationID
    """).fetchone()

    results.append({
        "file": file_path.name,
        "records": result[0],
        "earliest": result[1],
        "latest": result[2],
        "outside_month": result[3],
        "inside_month": result[4],
        "missing_zone": result[5],
        "unknown_zone": result[6],
    })

print("\n" + "=" * 110)
print("MONTHLY PARQUET VALIDATION")
print("=" * 110)

for row in results:
    print(
        f"{row['file']} | "
        f"Records: {row['records']:>10,} | "
        f"Outside month: {row['outside_month']:>8,} | "
        f"Missing zones: {row['missing_zone']:>7,} | "
        f"Unknown zones: {row['unknown_zone']:>7,}"
    )

print("\n" + "=" * 110)
print("OVERALL VALIDATION")
print("=" * 110)

print(f"Monthly files checked       : {len(results)}")
print(f"Total records               : {sum(r['records'] for r in results):,}")
print(f"Records outside file month  : {sum(r['outside_month'] for r in results):,}")
print(f"Records inside file month   : {sum(r['inside_month'] for r in results):,}")
print(f"Missing pickup zone records : {sum(r['missing_zone'] for r in results):,}")
print(f"Unknown pickup zone records : {sum(r['unknown_zone'] for r in results):,}")

print("\nDetailed timestamp ranges:")

for row in results:
    print(
        f"{row['file']}: "
        f"{row['earliest']} -> {row['latest']}"
    )

print("\nMonthly validation complete.")

con.close()
