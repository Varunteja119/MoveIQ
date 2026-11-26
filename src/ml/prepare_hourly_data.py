from pathlib import Path
import duckdb

RAW_DIR = Path("data/ml/raw")
OUTPUT_DIR = Path("data/ml/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "hourly_demand_2024_2025.parquet"

files = sorted(RAW_DIR.glob("yellow_tripdata_*.parquet"))

if len(files) != 24:
    raise RuntimeError(f"Expected 24 monthly files, found {len(files)}")

print(f"Found {len(files)} monthly files.")

con = duckdb.connect()

file_list = "[" + ", ".join(
    "'" + str(file.resolve()).replace("\\", "/") + "'"
    for file in files
) + "]"

print("\nAggregating hourly pickup demand...")

con.execute(f"""
    COPY (
        SELECT
            date_trunc('hour', tpep_pickup_datetime) AS pickup_hour,
            COUNT(*) AS trip_count
        FROM read_parquet({file_list}, union_by_name = true)
        WHERE
            tpep_pickup_datetime >= TIMESTAMP '2024-01-01 00:00:00'
            AND tpep_pickup_datetime < TIMESTAMP '2026-01-01 00:00:00'
        GROUP BY 1
        ORDER BY 1
    )
    TO '{output_file.resolve().as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

result = con.execute(f"""
    SELECT
        COUNT(*) AS hourly_records,
        MIN(pickup_hour) AS first_hour,
        MAX(pickup_hour) AS last_hour,
        SUM(trip_count) AS total_trips,
        AVG(trip_count) AS average_hourly_trips,
        MIN(trip_count) AS minimum_hourly_trips,
        MAX(trip_count) AS maximum_hourly_trips
    FROM read_parquet('{output_file.resolve().as_posix()}')
""").fetchdf()

print("\n" + "=" * 65)
print("HOURLY DATASET SUMMARY")
print("=" * 65)
print(result.to_string(index=False))

duplicate_count = con.execute(f"""
    SELECT COUNT(*) - COUNT(DISTINCT pickup_hour)
    FROM read_parquet('{output_file.resolve().as_posix()}')
""").fetchone()[0]

print(f"\nDuplicate hourly timestamps: {duplicate_count}")
print(f"Output file: {output_file}")

con.close()
