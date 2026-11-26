from pathlib import Path
from datetime import datetime
import re

import duckdb

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "ml" / "raw"
PROCESSED_DIR = ROOT / "data" / "ml" / "processed"

CITY_FILE = PROCESSED_DIR / "hourly_demand_2024_2025.parquet"
ZONE_FILE = PROCESSED_DIR / "zone_hourly_demand_2024_2025.parquet"

con = duckdb.connect()

print("=" * 75)
print("MOVEIQ - CITYWIDE VS ZONE DATA RECONCILIATION")
print("=" * 75)

# 1. Recalculate the source total using the intended monthly boundaries.
source_total = 0
source_missing_timestamp = 0
source_missing_zone = 0
monthly_results = []

for path in sorted(RAW_DIR.glob("yellow_tripdata_*.parquet")):

    match = re.search(r"yellow_tripdata_(\d{4})-(\d{2})", path.name)
    year, month = map(int, match.groups())

    start = datetime(year, month, 1)

    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)

    file_path = path.resolve().as_posix()

    result = con.execute(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (
                WHERE tpep_pickup_datetime IS NULL
            ) AS missing_timestamp,
            COUNT(*) FILTER (
                WHERE PULocationID IS NULL
            ) AS missing_zone,
            COUNT(*) FILTER (
                WHERE tpep_pickup_datetime IS NOT NULL
                  AND PULocationID IS NOT NULL
            ) AS usable_records
        FROM read_parquet('{file_path}')
        WHERE tpep_pickup_datetime >= TIMESTAMP '{start}'
          AND tpep_pickup_datetime < TIMESTAMP '{end}'
    """).fetchone()

    source_total += result[0]
    source_missing_timestamp += result[1]
    source_missing_zone += result[2]

    monthly_results.append((path.name, *result))

print("\nSOURCE TOTALS — MONTHLY BOUNDARIES APPLIED")
print(f"Records inside intended months : {source_total:,}")
print(f"Missing pickup timestamps      : {source_missing_timestamp:,}")
print(f"Missing pickup zone IDs        : {source_missing_zone:,}")

# 2. Existing citywide hourly dataset.
city_path = CITY_FILE.resolve().as_posix()

city = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT pickup_hour) AS unique_hours,
        SUM(trip_count) AS total_pickups,
        MIN(pickup_hour) AS first_hour,
        MAX(pickup_hour) AS last_hour
    FROM read_parquet('{city_path}')
""").fetchone()

print("\nEXISTING CITYWIDE HOURLY DATASET")
print(f"Rows            : {city[0]:,}")
print(f"Unique hours    : {city[1]:,}")
print(f"Total pickups   : {city[2]:,.0f}")
print(f"First hour      : {city[3]}")
print(f"Last hour       : {city[4]}")

# 3. New zone-level dataset.
zone_path = ZONE_FILE.resolve().as_posix()

zone = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT pickup_hour) AS unique_hours,
        COUNT(DISTINCT zone_id) AS zones,
        SUM(trip_count) AS total_pickups,
        MIN(pickup_hour) AS first_hour,
        MAX(pickup_hour) AS last_hour
    FROM read_parquet('{zone_path}')
""").fetchone()

print("\nNEW ZONE-HOURLY DATASET")
print(f"Rows            : {zone[0]:,}")
print(f"Unique hours    : {zone[1]:,}")
print(f"Zones           : {zone[2]:,}")
print(f"Total pickups   : {zone[3]:,.0f}")
print(f"First hour      : {zone[4]}")
print(f"Last hour       : {zone[5]}")

print("\nTOTAL RECONCILIATION")
print(f"Source minus zone dataset : {source_total - zone[3]:,}")
print(f"City minus zone dataset   : {city[2] - zone[3]:,.0f}")
print(f"Source minus city dataset : {source_total - city[2]:,.0f}")

# 4. Compare hourly timestamps.
print("\nTIMESTAMP DIFFERENCES")

missing_in_city = con.execute(f"""
    SELECT pickup_hour
    FROM read_parquet('{zone_path}')
    GROUP BY pickup_hour

    EXCEPT

    SELECT pickup_hour
    FROM read_parquet('{city_path}')
    ORDER BY pickup_hour
""").fetchdf()

missing_in_zone = con.execute(f"""
    SELECT pickup_hour
    FROM read_parquet('{city_path}')

    EXCEPT

    SELECT pickup_hour
    FROM read_parquet('{zone_path}')
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""").fetchdf()

print(f"Zone timestamps absent from city dataset: {len(missing_in_city)}")
print(missing_in_city.to_string(index=False))

print(f"\nCity timestamps absent from zone dataset: {len(missing_in_zone)}")
print(missing_in_zone.to_string(index=False))

# 5. Verify DST timestamps in the new calendar.
print("\nCHECKING SPRING DST HOURS")

dst_hours = [
    "2024-03-10 02:00:00",
    "2025-03-09 02:00:00",
]

for timestamp in dst_hours:
    count = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{zone_path}')
        WHERE pickup_hour = TIMESTAMP '{timestamp}'
    """).fetchone()[0]

    print(f"{timestamp}: {count:,} zone rows")

print("\nReconciliation diagnostic complete.")

con.close()
