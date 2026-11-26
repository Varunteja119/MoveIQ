from pathlib import Path
import calendar

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "ml" / "raw"
ZONE_LOOKUP = PROJECT_ROOT / "data" / "reference" / "taxi_zone_lookup.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "analytics" / "historical"
OUTPUT_FILE = OUTPUT_DIR / "zone_hourly_2024_2026.parquet"
TEMP_DIR = OUTPUT_DIR / "duckdb_temp"

START_DATE = "2024-01-01"
END_DATE = "2026-10-01"

EXPECTED_MONTHS = [
    f"{year}-{month:02d}"
    for year in range(2024, 2027)
    for month in range(1, 13)
    if (year, month) <= (2026, 9)
]

MEMORY_LIMIT = "4GB"
THREADS = 4


def main():
    files = sorted(RAW_DIR.glob("yellow_tripdata_*.parquet"))

    available_months = {
        file.stem.replace("yellow_tripdata_", "")
        for file in files
    }

    missing_months = sorted(set(EXPECTED_MONTHS) - available_months)
    unexpected_months = sorted(available_months - set(EXPECTED_MONTHS))

    if missing_months or unexpected_months:
        raise RuntimeError(
            f"Monthly file mismatch. Missing: {missing_months}; "
            f"Unexpected: {unexpected_months}"
        )

    if not ZONE_LOOKUP.exists():
        raise FileNotFoundError(ZONE_LOOKUP)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    raw_glob = (RAW_DIR / "yellow_tripdata_*.parquet").as_posix()
    zone_path = ZONE_LOOKUP.as_posix()
    output_path = OUTPUT_FILE.as_posix()
    temp_path = TEMP_DIR.as_posix()

    print("=" * 65)
    print("MOVEIQ HISTORICAL ANALYTICS AGGREGATION")
    print("=" * 65)
    print(f"Monthly files: {len(files)}")
    print(f"Coverage: {START_DATE} to 2026-09-30")
    print(f"Memory limit: {MEMORY_LIMIT}")
    print(f"Threads: {THREADS}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nStarting DuckDB aggregation...")

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET threads = {THREADS}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp_path}'")

    try:
        query = f"""
        COPY (
            WITH raw_trips AS (
                SELECT
                    CAST(tpep_pickup_datetime AS TIMESTAMP) AS pickup_ts,
                    CAST(tpep_dropoff_datetime AS TIMESTAMP) AS dropoff_ts,
                    CAST(PULocationID AS INTEGER) AS pickup_location_id,
                    CAST(DOLocationID AS INTEGER) AS dropoff_location_id,
                    CAST(fare_amount AS DOUBLE) AS fare_amount,
                    CAST(trip_distance AS DOUBLE) AS trip_distance
                FROM read_parquet(
                    '{raw_glob}',
                    union_by_name = true
                )
                WHERE tpep_pickup_datetime >= TIMESTAMP '{START_DATE}'
                  AND tpep_pickup_datetime < TIMESTAMP '{END_DATE}'
            ),

            measured_trips AS (
                SELECT
                    *,
                    date_diff(
                        'microsecond',
                        pickup_ts,
                        dropoff_ts
                    ) / 1000000.0 AS duration_seconds
                FROM raw_trips
            ),

            zone_roles AS (
                SELECT
                    date_trunc('hour', pickup_ts) AS pickup_hour,
                    'pickup' AS location_role,
                    pickup_location_id AS location_id,
                    fare_amount,
                    trip_distance,
                    duration_seconds
                FROM measured_trips

                UNION ALL

                SELECT
                    date_trunc('hour', pickup_ts) AS pickup_hour,
                    'dropoff' AS location_role,
                    dropoff_location_id AS location_id,
                    fare_amount,
                    trip_distance,
                    duration_seconds
                FROM measured_trips
            ),

            aggregated AS (
                SELECT
                    pickup_hour,
                    location_role,
                    location_id,

                    COUNT(*) AS total_trips,

                    COUNT(*) FILTER (
                        WHERE fare_amount >= 0
                          AND fare_amount <= 1000
                    ) AS valid_fare_trips,

                    COALESCE(
                        SUM(fare_amount) FILTER (
                            WHERE fare_amount >= 0
                              AND fare_amount <= 1000
                        ),
                        0
                    ) AS fare_sum,

                    COUNT(*) FILTER (
                        WHERE trip_distance >= 0
                          AND trip_distance <= 500
                    ) AS valid_distance_trips,

                    COALESCE(
                        SUM(trip_distance) FILTER (
                            WHERE trip_distance >= 0
                              AND trip_distance <= 500
                        ),
                        0
                    ) AS distance_sum,

                    COUNT(*) FILTER (
                        WHERE duration_seconds > 0
                          AND duration_seconds <= 86400
                    ) AS valid_duration_trips,

                    COALESCE(
                        SUM(duration_seconds) FILTER (
                            WHERE duration_seconds > 0
                              AND duration_seconds <= 86400
                        ),
                        0
                    ) AS duration_seconds_sum

                FROM zone_roles
                GROUP BY
                    pickup_hour,
                    location_role,
                    location_id
            )

            SELECT
                a.pickup_hour,
                CAST(a.pickup_hour AS DATE) AS pickup_date,
                EXTRACT(HOUR FROM a.pickup_hour)::INTEGER AS pickup_hour_of_day,
                a.location_role,
                a.location_id,

                COALESCE(z.Zone, 'Unknown') AS zone_name,
                COALESCE(z.Borough, 'Unknown') AS borough,

                a.total_trips,
                a.valid_fare_trips,
                a.fare_sum,
                a.valid_distance_trips,
                a.distance_sum,
                a.valid_duration_trips,
                a.duration_seconds_sum

            FROM aggregated a

            LEFT JOIN read_csv_auto('{zone_path}') z
                ON CAST(z.LocationID AS INTEGER) = a.location_id

        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """

        con.execute(query)

        print("\nAggregation completed. Checking output...")

        result = con.execute(
            f"""
            SELECT
                COUNT(*) AS aggregate_rows,
                MIN(pickup_hour) AS first_hour,
                MAX(pickup_hour) AS last_hour,
                SUM(total_trips) AS total_zone_trips,
                COUNT(DISTINCT location_id) AS distinct_zones
            FROM read_parquet('{output_path}')
            """
        ).fetchone()

        role_counts = con.execute(
            f"""
            SELECT
                location_role,
                COUNT(*) AS aggregate_rows,
                SUM(total_trips) AS total_trips
            FROM read_parquet('{output_path}')
            GROUP BY location_role
            ORDER BY location_role
            """
        ).fetchall()

        output_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 65)
        print("HISTORICAL AGGREGATION REPORT")
        print("=" * 65)
        print(f"Aggregate rows: {result[0]:,}")
        print(f"First hour: {result[1]}")
        print(f"Last hour: {result[2]}")
        print(f"Zone IDs: {result[4]}")
        print(f"Total zone trips across both roles: {result[3]:,}")
        print(f"Output size: {output_size:,.2f} MB")

        print("\nBreakdown by location role:")
        for role, rows, trips in role_counts:
            print(
                f"{role.title():8} | "
                f"{rows:>12,} aggregates | "
                f"{trips:>15,} trips"
            )

        print("\nOutput validation:")
        if result[0] == 0:
            raise RuntimeError("Historical aggregation produced no rows.")

        if result[1] is None or result[2] is None:
            raise RuntimeError("Historical date coverage is missing.")

        if len(role_counts) != 2:
            raise RuntimeError("Pickup/dropoff aggregation is incomplete.")

        print("PASS - Historical aggregate is readable.")
        print("PASS - Pickup and dropoff aggregates exist.")
        print(f"\nSaved: {OUTPUT_FILE}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
