from pathlib import Path
import duckdb

FEATURE_FILE = Path("data/ml/processed/zone_hourly_demand_features.parquet")
con = duckdb.connect()

path = FEATURE_FILE.resolve().as_posix()

print("=" * 75)
print("ZONE FEATURE VALIDATION")
print("=" * 75)

checks = {}

checks["Negative demand values"] = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{path}')
    WHERE trip_count < 0
""").fetchone()[0]

checks["Duplicate zone-hour records"] = con.execute(f"""
    SELECT COUNT(*) FROM (
        SELECT pickup_hour, zone_id
        FROM read_parquet('{path}')
        GROUP BY 1, 2
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

checks["Incorrect lag_1 values"] = con.execute(f"""
    WITH expected AS (
        SELECT
            zone_id,
            pickup_hour,
            LAG(trip_count) OVER (
                PARTITION BY zone_id ORDER BY pickup_hour
            ) AS expected_lag
        FROM read_parquet('{path}')
    )
    SELECT COUNT(*)
    FROM expected e
    JOIN read_parquet('{path}') f
      USING (zone_id, pickup_hour)
    WHERE e.expected_lag IS DISTINCT FROM f.lag_1
""").fetchone()[0]

checks["Incorrect lag_24 values"] = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{path}') f
    LEFT JOIN read_parquet('{path}') p
      ON f.zone_id = p.zone_id
     AND p.pickup_hour = f.pickup_hour - INTERVAL '24 hours'
    WHERE f.lag_24 IS DISTINCT FROM p.trip_count
""").fetchone()[0]

checks["Incorrect lag_168 values"] = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{path}') f
    LEFT JOIN read_parquet('{path}') p
      ON f.zone_id = p.zone_id
     AND p.pickup_hour = f.pickup_hour - INTERVAL '168 hours'
    WHERE f.lag_168 IS DISTINCT FROM p.trip_count
""").fetchone()[0]

checks["Incorrect 24-hour rolling averages"] = con.execute(f"""
    WITH expected AS (
        SELECT
            zone_id,
            pickup_hour,
            AVG(trip_count) OVER (
                PARTITION BY zone_id
                ORDER BY pickup_hour
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                AND INTERVAL '1 hour' PRECEDING
            ) AS expected_mean
        FROM read_parquet('{path}')
    )
    SELECT COUNT(*)
    FROM expected e
    JOIN read_parquet('{path}') f
      USING (zone_id, pickup_hour)
    WHERE ABS(e.expected_mean - f.rolling_mean_24) > 0.000001
       OR (e.expected_mean IS NULL AND f.rolling_mean_24 IS NOT NULL)
       OR (e.expected_mean IS NOT NULL AND f.rolling_mean_24 IS NULL)
""").fetchone()[0]

print("\nVALIDATION RESULTS")
print("-" * 75)

for name, value in checks.items():
    status = "PASS" if value == 0 else "FAIL"
    print(f"{name:<42} {value:>8,}  {status}")

print("\nDST TRANSITION CHECK")
print("-" * 75)

print(con.execute(f"""
    SELECT
        pickup_hour,
        zone_id,
        trip_count,
        lag_1,
        lag_24,
        lag_168
    FROM read_parquet('{path}')
    WHERE pickup_hour BETWEEN
        TIMESTAMP '2024-03-10 00:00:00'
        AND TIMESTAMP '2024-03-10 04:00:00'
      AND zone_id = 161

    UNION ALL

    SELECT
        pickup_hour,
        zone_id,
        trip_count,
        lag_1,
        lag_24,
        lag_168
    FROM read_parquet('{path}')
    WHERE pickup_hour BETWEEN
        TIMESTAMP '2025-03-09 00:00:00'
        AND TIMESTAMP '2025-03-09 04:00:00'
      AND zone_id = 161

    ORDER BY pickup_hour
""").fetchdf().to_string(index=False))

con.close()

if any(checks.values()):
    raise RuntimeError("Feature validation failed.")

print("\nSUCCESS: All feature calculations passed validation.")
