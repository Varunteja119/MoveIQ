from pathlib import Path
import duckdb

FEATURE_FILE = Path("data/ml/processed/zone_hourly_demand_features.parquet")

con = duckdb.connect()
path = FEATURE_FILE.resolve().as_posix()

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE active_zones AS
    SELECT zone_id
    FROM read_parquet('{path}')
    GROUP BY zone_id
    HAVING SUM(trip_count) > 0
""")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE base AS
    SELECT f.*
    FROM read_parquet('{path}') f
    INNER JOIN active_zones z USING (zone_id)
    WHERE f.lag_24 IS NOT NULL
      AND f.lag_168 IS NOT NULL
      AND f.rolling_mean_24 IS NOT NULL
      AND f.rolling_mean_168 IS NOT NULL
""")

con.execute("""
    CREATE OR REPLACE TEMP TABLE complete_origins AS
    SELECT
        origin.zone_id,
        origin.pickup_hour AS origin_timestamp,
        COUNT(target.pickup_hour) AS target_count,
        COUNT(target.lag_24) AS target_lag24_count,
        COUNT(target.lag_168) AS target_lag168_count
    FROM base origin
    JOIN base target
      ON target.zone_id = origin.zone_id
     AND target.pickup_hour > origin.pickup_hour
     AND target.pickup_hour <= origin.pickup_hour + INTERVAL '24 hours'
    WHERE DATE_DIFF(
        'hour',
        TIMESTAMP '2024-01-01 00:00:00',
        origin.pickup_hour
    ) % 6 = 0
    GROUP BY origin.zone_id, origin.pickup_hour
    HAVING COUNT(target.pickup_hour) = 24
       AND COUNT(target.lag_24) = 24
       AND COUNT(target.lag_168) = 24
""")

print("=" * 75)
print("CORRECTED DIRECT FORECAST CAPACITY")
print("=" * 75)

print("Active zones:", con.execute(
    "SELECT COUNT(*) FROM active_zones"
).fetchone()[0])

print("\nSplit estimates:")

print(con.execute("""
    WITH splits AS (
        SELECT
            zone_id,
            origin_timestamp,
            CASE
                WHEN origin_timestamp >= TIMESTAMP '2024-01-08 00:00:00'
                 AND origin_timestamp + INTERVAL '24 hours'
                     <= TIMESTAMP '2025-05-28 20:00:00'
                THEN 'train'

                WHEN origin_timestamp >= TIMESTAMP '2025-05-28 21:00:00'
                 AND origin_timestamp + INTERVAL '24 hours'
                     <= TIMESTAMP '2025-09-14 09:00:00'
                THEN 'validation'

                WHEN origin_timestamp >= TIMESTAMP '2025-09-14 10:00:00'
                 AND origin_timestamp + INTERVAL '24 hours'
                     <= TIMESTAMP '2025-12-31 23:00:00'
                THEN 'test'
            END AS dataset_split
        FROM complete_origins
    )
    SELECT
        dataset_split,
        COUNT(*) AS zone_origins,
        COUNT(DISTINCT zone_id) AS zones,
        COUNT(DISTINCT origin_timestamp) AS forecast_origins,
        COUNT(*) * 24 AS estimated_samples
    FROM splits
    WHERE dataset_split IS NOT NULL
    GROUP BY dataset_split
    ORDER BY
        CASE dataset_split
            WHEN 'train' THEN 1
            WHEN 'validation' THEN 2
            WHEN 'test' THEN 3
        END
""").fetchdf().to_string(index=False))

con.close()
