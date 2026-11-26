from pathlib import Path
import duckdb

INPUT_FILE = Path("data/ml/processed/zone_hourly_demand_features.parquet")
OUTPUT_DIR = Path("data/ml/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2025-05-28 20:00:00"
VALIDATION_START = "2025-05-28 21:00:00"
VALIDATION_END = "2025-09-14 09:00:00"
TEST_START = "2025-09-14 10:00:00"

con = duckdb.connect()

source = INPUT_FILE.resolve().as_posix()

print("=" * 75)
print("MOVEIQ ZONE-LEVEL CHRONOLOGICAL DATA SPLIT")
print("=" * 75)

print("\nIdentifying zones with historical demand...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE active_zones AS
    SELECT zone_id
    FROM read_parquet('{source}')
    GROUP BY zone_id
    HAVING SUM(trip_count) > 0
""")

active_zone_count = con.execute("""
    SELECT COUNT(*) FROM active_zones
""").fetchone()[0]

print(f"Active zones: {active_zone_count}")

print("\nFiltering rows with complete historical features...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE model_data AS
    SELECT f.*
    FROM read_parquet('{source}') f
    INNER JOIN active_zones z USING (zone_id)
    WHERE
        f.lag_1 IS NOT NULL
        AND f.lag_24 IS NOT NULL
        AND f.lag_168 IS NOT NULL
        AND f.rolling_mean_24 IS NOT NULL
        AND f.rolling_mean_168 IS NOT NULL
""")

total_rows = con.execute("""
    SELECT COUNT(*) FROM model_data
""").fetchone()[0]

print(f"Usable zone-hour records: {total_rows:,}")

splits = [
    (
        "TRAIN",
        f"pickup_hour <= TIMESTAMP '{TRAIN_END}'",
        "zone_train.parquet"
    ),
    (
        "VALIDATION",
        f"pickup_hour >= TIMESTAMP '{VALIDATION_START}' "
        f"AND pickup_hour <= TIMESTAMP '{VALIDATION_END}'",
        "zone_validation.parquet"
    ),
    (
        "TEST",
        f"pickup_hour >= TIMESTAMP '{TEST_START}'",
        "zone_test.parquet"
    )
]

print("\nCreating chronological datasets...")

for name, condition, filename in splits:
    output_file = OUTPUT_DIR / filename
    output_path = output_file.resolve().as_posix()

    con.execute(f"""
        COPY (
            SELECT *
            FROM model_data
            WHERE {condition}
            ORDER BY pickup_hour, zone_id
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    summary = con.execute(f"""
        SELECT
            COUNT(*) AS records,
            COUNT(DISTINCT zone_id) AS zones,
            COUNT(DISTINCT pickup_hour) AS hours,
            MIN(pickup_hour) AS start_time,
            MAX(pickup_hour) AS end_time,
            SUM(trip_count) AS pickups
        FROM read_parquet('{output_path}')
    """).fetchdf()

    print(f"\n{name}")
    print(summary.to_string(index=False))
    print(f"Saved: {output_file}")

print("\n" + "=" * 75)
print("FINAL SPLIT VALIDATION")
print("=" * 75)

train_count = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet(
        '{(OUTPUT_DIR / "zone_train.parquet").resolve().as_posix()}'
    )
""").fetchone()[0]

validation_count = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet(
        '{(OUTPUT_DIR / "zone_validation.parquet").resolve().as_posix()}'
    )
""").fetchone()[0]

test_count = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet(
        '{(OUTPUT_DIR / "zone_test.parquet").resolve().as_posix()}'
    )
""").fetchone()[0]

print(f"Train records      : {train_count:,}")
print(f"Validation records : {validation_count:,}")
print(f"Test records       : {test_count:,}")
print(f"Total split records: {train_count + validation_count + test_count:,}")
print(f"Usable records     : {total_rows:,}")

if train_count + validation_count + test_count != total_rows:
    raise RuntimeError("Split record counts do not reconcile.")

for filename in [
    "zone_train.parquet",
    "zone_validation.parquet",
    "zone_test.parquet"
]:
    path = (OUTPUT_DIR / filename).resolve().as_posix()

    zone_count = con.execute(f"""
        SELECT COUNT(DISTINCT zone_id)
        FROM read_parquet('{path}')
    """).fetchone()[0]

    if zone_count != active_zone_count:
        raise RuntimeError(
            f"{filename} contains {zone_count} zones; "
            f"expected {active_zone_count}"
        )

print("\nSUCCESS: Chronological zone datasets created and validated.")
print("Original citywide datasets were not modified.")

con.close()
