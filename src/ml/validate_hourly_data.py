from pathlib import Path
import duckdb

DATA_FILE = Path("data/ml/processed/hourly_demand_2024_2025.parquet")

con = duckdb.connect()

query = f"""
WITH expected_hours AS (
    SELECT generate_series AS pickup_hour
    FROM generate_series(
        TIMESTAMP '2024-01-01 00:00:00',
        TIMESTAMP '2025-12-31 23:00:00',
        INTERVAL 1 HOUR
    )
),
actual_hours AS (
    SELECT pickup_hour
    FROM read_parquet('{DATA_FILE.resolve().as_posix()}')
)
SELECT e.pickup_hour
FROM expected_hours e
LEFT JOIN actual_hours a
    ON e.pickup_hour = a.pickup_hour
WHERE a.pickup_hour IS NULL
ORDER BY e.pickup_hour
"""

missing = con.execute(query).fetchdf()

print("=" * 60)
print("HOURLY TIMELINE VALIDATION")
print("=" * 60)

print(f"Missing timestamps: {len(missing)}")

if not missing.empty:
    print("\nMissing hours:")
    print(missing.to_string(index=False))

print("\nDaily record counts around missing timestamps:")

if not missing.empty:
    for timestamp in missing["pickup_hour"]:
        date = timestamp.date()
        daily = con.execute(f"""
            SELECT
                CAST(pickup_hour AS DATE) AS date,
                COUNT(*) AS hourly_records,
                SUM(trip_count) AS trips
            FROM read_parquet('{DATA_FILE.resolve().as_posix()}')
            WHERE CAST(pickup_hour AS DATE)
                BETWEEN DATE '{date}' - INTERVAL 1 DAY
                AND DATE '{date}' + INTERVAL 1 DAY
            GROUP BY 1
            ORDER BY 1
        """).fetchdf()

        print(f"\nAround {date}:")
        print(daily.to_string(index=False))

con.close()
