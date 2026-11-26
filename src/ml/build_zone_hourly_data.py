from pathlib import Path
import duckdb

RAW_DIR = Path("data/ml/raw")
PROCESSED_DIR = Path("data/ml/processed")
REFERENCE_DIR = Path("data/reference")

OUTPUT_FILE = PROCESSED_DIR / "zone_hourly_demand_2024_2025.parquet"
CITY_FILE = PROCESSED_DIR / "hourly_demand_2024_2025.parquet"
LOOKUP_FILE = REFERENCE_DIR / "taxi_zone_lookup.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("yellow_tripdata_*.parquet"))

if len(files) != 24:
    raise RuntimeError(f"Expected 24 monthly files, found {len(files)}")

for path in [CITY_FILE, LOOKUP_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

file_list = "[" + ", ".join(
    "'" + str(file.resolve()).replace("\\", "/") + "'"
    for file in files
) + "]"

city_path = CITY_FILE.resolve().as_posix()
lookup_path = LOOKUP_FILE.resolve().as_posix()
output_path = OUTPUT_FILE.resolve().as_posix()

con = duckdb.connect()

print("=" * 75)
print("REBUILDING ZONE-HOURLY DEMAND DATA")
print("=" * 75)

print("\nAggregating pickup demand using the citywide date filter...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE zone_counts AS
    SELECT
        DATE_TRUNC('hour', tpep_pickup_datetime) AS pickup_hour,
        PULocationID::INTEGER AS zone_id,
        COUNT(*)::BIGINT AS trip_count
    FROM read_parquet({file_list}, union_by_name = true)
    WHERE
        tpep_pickup_datetime >= TIMESTAMP '2024-01-01 00:00:00'
        AND tpep_pickup_datetime < TIMESTAMP '2026-01-01 00:00:00'
        AND PULocationID IS NOT NULL
    GROUP BY 1, 2
""")

print("Using actual citywide timestamps and official taxi zones...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE actual_hours AS
    SELECT pickup_hour
    FROM read_parquet('{city_path}')
""")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE official_zones AS
    SELECT DISTINCT
        LocationID::INTEGER AS zone_id
    FROM read_csv_auto('{lookup_path}')
    WHERE LocationID IS NOT NULL
""")

print("\nBuilding complete zone-hour grid...")

con.execute(f"""
    COPY (
        SELECT
            h.pickup_hour,
            z.zone_id,
            COALESCE(c.trip_count, 0)::BIGINT AS trip_count
        FROM actual_hours h
        CROSS JOIN official_zones z
        LEFT JOIN zone_counts c
            ON h.pickup_hour = c.pickup_hour
            AND z.zone_id = c.zone_id
        ORDER BY h.pickup_hour, z.zone_id
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nValidating rebuilt dataset...")

summary = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT pickup_hour) AS unique_hours,
        COUNT(DISTINCT zone_id) AS unique_zones,
        SUM(trip_count) AS total_pickups,
        AVG(trip_count) AS average_zone_hourly_demand,
        MIN(trip_count) AS minimum_demand,
        MAX(trip_count) AS maximum_demand,
        SUM(CASE WHEN trip_count = 0 THEN 1 ELSE 0 END) AS zero_demand_rows
    FROM read_parquet('{output_path}')
""").fetchdf()

city_total = con.execute(f"""
    SELECT SUM(trip_count)
    FROM read_parquet('{city_path}')
""").fetchone()[0]

hourly_mismatches = con.execute(f"""
    WITH zone_hourly AS (
        SELECT pickup_hour, SUM(trip_count) AS zone_pickups
        FROM read_parquet('{output_path}')
        GROUP BY pickup_hour
    )
    SELECT COUNT(*)
    FROM zone_hourly z
    JOIN read_parquet('{city_path}') c USING (pickup_hour)
    WHERE z.zone_pickups != c.trip_count
""").fetchone()[0]

print("\n" + "=" * 75)
print("ZONE-HOURLY DATASET SUMMARY")
print("=" * 75)
print(summary.to_string(index=False))

zone_total = int(summary.iloc[0]["total_pickups"])

print(f"\nCitywide pickup total       : {city_total:,}")
print(f"Zone-level pickup total    : {zone_total:,}")
print(f"Difference                 : {int(city_total) - zone_total:,}")
print(f"Hourly reconciliation errors: {hourly_mismatches:,}")
print(f"Expected grid rows         : {int(summary.iloc[0]['unique_hours']) * int(summary.iloc[0]['unique_zones']):,}")
print(f"Output file                : {OUTPUT_FILE}")

if int(city_total) != zone_total or hourly_mismatches != 0:
    raise RuntimeError("Reconciliation failed. Do not proceed to model training.")

print("\nSUCCESS: Every zone-hour total reconciles with the citywide dataset.")

con.close()
