from pathlib import Path
import duckdb

INPUT_FILE = Path("data/ml/processed/hourly_demand_features.parquet")
TEMP_FILE = Path("data/ml/processed/hourly_demand_features_temp.parquet")

con = duckdb.connect()

source = INPUT_FILE.resolve().as_posix()
destination = TEMP_FILE.resolve().as_posix()

print("Correcting lag_1 using previous actual observation...")

con.execute(f"""
    COPY (
        SELECT
            * EXCLUDE (lag_1),
            LAG(trip_count) OVER (
                ORDER BY pickup_hour
            ) AS lag_1
        FROM read_parquet('{source}')
        ORDER BY pickup_hour
    )
    TO '{destination}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

con.close()

TEMP_FILE.replace(INPUT_FILE)

con = duckdb.connect()
result = con.execute(f"""
    SELECT
        COUNT(*) AS total_records,
        COUNT(lag_1) AS valid_lag_1,
        COUNT(*) - COUNT(lag_1) AS missing_lag_1
    FROM read_parquet('{source}')
""").fetchdf()

print("\nUpdated lag_1 validation:")
print(result.to_string(index=False))

con.close()
print("\nFeature dataset updated successfully.")
