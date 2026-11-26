from pathlib import Path
import duckdb

PROCESSED_DIR = Path("data/ml/processed")
REFERENCE_DIR = Path("data/reference")

INPUT_FILE = PROCESSED_DIR / "zone_hourly_demand_2024_2025.parquet"
LOOKUP_FILE = REFERENCE_DIR / "taxi_zone_lookup.csv"
OUTPUT_FILE = PROCESSED_DIR / "zone_hourly_demand_features.parquet"

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

if not LOOKUP_FILE.exists():
    raise FileNotFoundError(LOOKUP_FILE)

input_path = INPUT_FILE.resolve().as_posix()
lookup_path = LOOKUP_FILE.resolve().as_posix()
output_path = OUTPUT_FILE.resolve().as_posix()

con = duckdb.connect()

print("=" * 75)
print("MOVEIQ ZONE-LEVEL FEATURE ENGINEERING")
print("=" * 75)

print("\nLoading zone demand and geographic metadata...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE base AS
    SELECT
        d.pickup_hour,
        d.zone_id,
        d.trip_count,
        COALESCE(l.Borough, 'Unknown') AS borough,
        COALESCE(l.Zone, 'Unknown') AS zone,
        COALESCE(l.service_zone, 'Unknown') AS service_zone,
        EXTRACT(HOUR FROM d.pickup_hour)::INTEGER AS hour,
        ((DAYOFWEEK(d.pickup_hour) + 5) % 7)::INTEGER AS day_of_week,
        CASE
            WHEN DAYOFWEEK(d.pickup_hour) IN (0, 6) THEN 1
            ELSE 0
        END AS is_weekend,
        EXTRACT(MONTH FROM d.pickup_hour)::INTEGER AS month
    FROM read_parquet('{input_path}') d
    LEFT JOIN read_csv_auto('{lookup_path}') l
        ON d.zone_id = l.LocationID
""")

print("Calculating zone-specific historical lags...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE lag_features AS
    SELECT
        *,
        LAG(trip_count, 1) OVER (
            PARTITION BY zone_id
            ORDER BY pickup_hour
        ) AS lag_1
    FROM base
""")

print("Calculating exact 24-hour and 168-hour lags...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE exact_lags AS
    SELECT
        current.*,
        previous_24.trip_count AS lag_24,
        previous_168.trip_count AS lag_168
    FROM lag_features current
    LEFT JOIN base previous_24
        ON current.zone_id = previous_24.zone_id
        AND previous_24.pickup_hour =
            current.pickup_hour - INTERVAL '24 hours'
    LEFT JOIN base previous_168
        ON current.zone_id = previous_168.zone_id
        AND previous_168.pickup_hour =
            current.pickup_hour - INTERVAL '168 hours'
""")

print("Calculating timestamp-based rolling averages...")

con.execute(f"""
    COPY (
        SELECT
            *,
            AVG(trip_count) OVER (
                PARTITION BY zone_id
                ORDER BY pickup_hour
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                AND INTERVAL '1 hour' PRECEDING
            ) AS rolling_mean_24,
            AVG(trip_count) OVER (
                PARTITION BY zone_id
                ORDER BY pickup_hour
                RANGE BETWEEN INTERVAL '168 hours' PRECEDING
                AND INTERVAL '1 hour' PRECEDING
            ) AS rolling_mean_168
        FROM exact_lags
        ORDER BY pickup_hour, zone_id
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nValidating feature dataset...")

summary = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT zone_id) AS unique_zones,
        COUNT(DISTINCT pickup_hour) AS unique_hours,
        SUM(trip_count) AS total_pickups,
        COUNT(*) FILTER (WHERE lag_1 IS NULL) AS null_lag_1,
        COUNT(*) FILTER (WHERE lag_24 IS NULL) AS null_lag_24,
        COUNT(*) FILTER (WHERE lag_168 IS NULL) AS null_lag_168,
        COUNT(*) FILTER (WHERE rolling_mean_24 IS NULL) AS null_rolling_24,
        COUNT(*) FILTER (WHERE rolling_mean_168 IS NULL) AS null_rolling_168
    FROM read_parquet('{output_path}')
""").fetchdf()

print("\n" + "=" * 75)
print("ZONE FEATURE DATASET SUMMARY")
print("=" * 75)
print(summary.to_string(index=False))

print("\nFeature sample:")

print(con.execute(f"""
    SELECT
        pickup_hour,
        zone_id,
        borough,
        trip_count,
        hour,
        day_of_week,
        is_weekend,
        month,
        lag_1,
        lag_24,
        lag_168,
        rolling_mean_24,
        rolling_mean_168
    FROM read_parquet('{output_path}')
    WHERE zone_id = 161
    ORDER BY pickup_hour
    LIMIT 10
""").fetchdf().to_string(index=False))

print(f"\nOutput file: {OUTPUT_FILE}")
print("\nFeature engineering completed.")

con.close()
