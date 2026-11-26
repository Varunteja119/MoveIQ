import duckdb

con = duckdb.connect()

path = "data/ml/processed/direct_forecasting_samples.parquet"

print("=" * 70)
print("CITYWIDE DIRECT DATASET SUMMARY")
print("=" * 70)

print(con.execute(f"""
    SELECT
        dataset_split,
        COUNT(*) AS samples,
        COUNT(DISTINCT origin_timestamp) AS forecast_origins,
        COUNT(DISTINCT hours_ahead) AS forecast_horizons,
        MIN(origin_timestamp) AS first_origin,
        MAX(origin_timestamp) AS last_origin
    FROM read_parquet('{path}')
    GROUP BY dataset_split
    ORDER BY
        CASE dataset_split
            WHEN 'train' THEN 1
            WHEN 'validation' THEN 2
            WHEN 'test' THEN 3
            ELSE 4
        END
""").fetchdf().to_string(index=False))

print("\nForecast horizon distribution:")

print(con.execute(f"""
    SELECT
        hours_ahead,
        COUNT(*) AS samples
    FROM read_parquet('{path}')
    GROUP BY hours_ahead
    ORDER BY hours_ahead
""").fetchdf().to_string(index=False))

print("\nSample records:")

print(con.execute(f"""
    SELECT *
    FROM read_parquet('{path}')
    ORDER BY origin_timestamp
    LIMIT 5
""").fetchdf().to_string(index=False))

con.close()
