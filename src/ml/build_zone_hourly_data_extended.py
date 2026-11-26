from pathlib import Path
import re
import duckdb

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data/ml/raw"
PROCESSED_DIR = ROOT / "data/ml/processed"
REFERENCE_FILE = ROOT / "data/reference/taxi_zone_lookup.csv"

CITY_FILE = PROCESSED_DIR / "hourly_demand_2024_2026.parquet"
OUTPUT_FILE = PROCESSED_DIR / "zone_hourly_demand_2024_2026.parquet"


def main():
    for path in [CITY_FILE, REFERENCE_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    pattern = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet$")
    files = sorted(
        path for path in RAW_DIR.glob("yellow_tripdata_*.parquet")
        if pattern.fullmatch(path.name)
    )

    if not files:
        raise RuntimeError("No TLC monthly files found.")

    for path in files:
        with path.open("rb") as file:
            if file.read(4) != b"PAR1":
                raise RuntimeError(f"Invalid Parquet file: {path.name}")

    file_list = "[" + ", ".join(
        "'" + str(path.resolve()).replace("\\", "/") + "'"
        for path in files
    ) + "]"

    city_path = CITY_FILE.resolve().as_posix()
    lookup_path = REFERENCE_FILE.resolve().as_posix()
    output_path = OUTPUT_FILE.resolve().as_posix()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("MOVEIQ — EXTENDED ZONE-HOURLY DATA BUILDER")
    print("=" * 75)
    print(f"Monthly files: {len(files)}")
    print(f"Citywide source: {CITY_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")

    print("\nStep 1: Aggregating pickup demand by hour and zone...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE zone_counts AS
        SELECT
            DATE_TRUNC('hour', tpep_pickup_datetime) AS pickup_hour,
            PULocationID::INTEGER AS zone_id,
            COUNT(*)::BIGINT AS trip_count
        FROM read_parquet({file_list}, union_by_name = true)
        WHERE
            tpep_pickup_datetime >= (
                SELECT MIN(pickup_hour)
                FROM read_parquet('{city_path}')
            )
            AND tpep_pickup_datetime < (
                SELECT MAX(pickup_hour) + INTERVAL '1 hour'
                FROM read_parquet('{city_path}')
            )
            AND PULocationID IS NOT NULL
        GROUP BY 1, 2
    """)

    print("Step 2: Loading actual citywide timestamps and official zones...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE actual_hours AS
        SELECT pickup_hour
        FROM read_parquet('{city_path}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE official_zones AS
        SELECT DISTINCT LocationID::INTEGER AS zone_id
        FROM read_csv_auto('{lookup_path}')
        WHERE LocationID IS NOT NULL
    """)

    hour_count = con.execute(
        "SELECT COUNT(*) FROM actual_hours"
    ).fetchone()[0]

    zone_count = con.execute(
        "SELECT COUNT(*) FROM official_zones"
    ).fetchone()[0]

    print(f"Actual hours: {hour_count:,}")
    print(f"Official zones: {zone_count:,}")
    print(f"Expected grid rows: {hour_count * zone_count:,}")

    print("\nStep 3: Building complete zone-hour grid...")

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

    print("\nStep 4: Validating the extended dataset...")

    summary = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT pickup_hour) AS unique_hours,
            COUNT(DISTINCT zone_id) AS unique_zones,
            SUM(trip_count) AS total_pickups,
            AVG(trip_count) AS average_zone_hourly_demand,
            MIN(trip_count) AS minimum_demand,
            MAX(trip_count) AS maximum_demand,
            COUNT(*) FILTER (WHERE trip_count = 0) AS zero_demand_rows
        FROM read_parquet('{output_path}')
    """).fetchdf()

    city_total = con.execute(f"""
        SELECT SUM(trip_count)
        FROM read_parquet('{city_path}')
    """).fetchone()[0]

    zone_total = int(summary.iloc[0]["total_pickups"])

    hourly_mismatches = con.execute(f"""
        WITH zone_hourly AS (
            SELECT pickup_hour, SUM(trip_count) AS zone_pickups
            FROM read_parquet('{output_path}')
            GROUP BY pickup_hour
        )
        SELECT COUNT(*)
        FROM zone_hourly z
        FULL OUTER JOIN read_parquet('{city_path}') c
            USING (pickup_hour)
        WHERE
            z.pickup_hour IS NULL
            OR c.pickup_hour IS NULL
            OR z.zone_pickups != c.trip_count
    """).fetchone()[0]

    expected_rows = hour_count * zone_count

    print("\n" + "=" * 75)
    print("EXTENDED ZONE-HOURLY DATASET SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))

    print(f"\nCitywide pickup total : {int(city_total):,}")
    print(f"Zone-level pickup total: {zone_total:,}")
    print(f"Difference            : {int(city_total) - zone_total:,}")
    print(f"Hourly mismatches     : {hourly_mismatches:,}")
    print(f"Expected grid rows    : {expected_rows:,}")

    if int(summary.iloc[0]["total_rows"]) != expected_rows:
        raise RuntimeError("Zone-hour grid row count does not match expectation.")

    if int(city_total) != zone_total or hourly_mismatches != 0:
        raise RuntimeError(
            "Citywide and zone-level demand do not reconcile. "
            "Do not proceed to feature engineering."
        )

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")
    print("\nSUCCESS: Extended zone data reconciles with citywide data.")

    con.close()


if __name__ == "__main__":
    main()