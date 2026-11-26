from pathlib import Path
import duckdb

INPUT_FILE = Path("data/ml/processed/hourly_demand_2024_2025.parquet")
OUTPUT_DIR = Path("data/ml/processed")
OUTPUT_FILE = OUTPUT_DIR / "hourly_demand_features.parquet"

con = duckdb.connect()

source = INPUT_FILE.resolve().as_posix()
destination = OUTPUT_FILE.resolve().as_posix()

print("Building time-based forecasting features...")

con.execute(f"""
    COPY (
        WITH base AS (
            SELECT
                pickup_hour,
                CAST(trip_count AS BIGINT) AS trip_count,
                EXTRACT(HOUR FROM pickup_hour)::INTEGER AS hour,
                EXTRACT(DOW FROM pickup_hour)::INTEGER AS day_of_week,
                CASE
                    WHEN EXTRACT(DOW FROM pickup_hour) IN (0, 6)
                    THEN 1 ELSE 0
                END AS is_weekend,
                EXTRACT(MONTH FROM pickup_hour)::INTEGER AS month
            FROM read_parquet('{source}')
        ),

        features AS (
            SELECT
                b.*,

                lag1.trip_count AS lag_1,
                lag24.trip_count AS lag_24,
                lag168.trip_count AS lag_168,

                rolling24.rolling_mean_24,
                rolling168.rolling_mean_168

            FROM base b

            LEFT JOIN base lag1
                ON lag1.pickup_hour = b.pickup_hour - INTERVAL 1 HOUR

            LEFT JOIN base lag24
                ON lag24.pickup_hour = b.pickup_hour - INTERVAL 24 HOURS

            LEFT JOIN base lag168
                ON lag168.pickup_hour = b.pickup_hour - INTERVAL 168 HOURS

            LEFT JOIN LATERAL (
                SELECT AVG(previous.trip_count) AS rolling_mean_24
                FROM base previous
                WHERE previous.pickup_hour < b.pickup_hour
                  AND previous.pickup_hour >= b.pickup_hour - INTERVAL 24 HOURS
            ) rolling24 ON TRUE

            LEFT JOIN LATERAL (
                SELECT AVG(previous.trip_count) AS rolling_mean_168
                FROM base previous
                WHERE previous.pickup_hour < b.pickup_hour
                  AND previous.pickup_hour >= b.pickup_hour - INTERVAL 168 HOURS
            ) rolling168 ON TRUE
        )

        SELECT *
        FROM features
        ORDER BY pickup_hour
    )
    TO '{destination}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

summary = con.execute(f"""
    SELECT
        COUNT(*) AS total_records,
        COUNT(lag_1) AS valid_lag_1,
        COUNT(lag_24) AS valid_lag_24,
        COUNT(lag_168) AS valid_lag_168,
        COUNT(rolling_mean_24) AS valid_rolling_24,
        COUNT(rolling_mean_168) AS valid_rolling_168,
        COUNT(*) FILTER (
            WHERE lag_1 IS NULL
               OR lag_24 IS NULL
               OR lag_168 IS NULL
               OR rolling_mean_24 IS NULL
               OR rolling_mean_168 IS NULL
        ) AS records_with_missing_features
    FROM read_parquet('{destination}')
""").fetchdf()

print("\n" + "=" * 65)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 65)
print(summary.to_string(index=False))

print("\nSample records:")
sample = con.execute(f"""
    SELECT *
    FROM read_parquet('{destination}')
    ORDER BY pickup_hour
    LIMIT 5
""").fetchdf()

print(sample.to_string(index=False))

print(f"\nSaved features to: {OUTPUT_FILE}")

con.close()
