from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]

HISTORY_FILE = ROOT / "data/ml/processed/hourly_demand_2024_2025.parquet"
SPLIT_DIR = ROOT / "data/ml/processed"

TRAIN_FILE = SPLIT_DIR / "train.parquet"
VALIDATION_FILE = SPLIT_DIR / "validation.parquet"

OUTPUT_FILE = SPLIT_DIR / "direct_forecasting_samples.parquet"

con = duckdb.connect()

history_path = HISTORY_FILE.resolve().as_posix()
train_path = TRAIN_FILE.resolve().as_posix()
validation_path = VALIDATION_FILE.resolve().as_posix()
output_path = OUTPUT_FILE.resolve().as_posix()

print("Loading historical demand and split boundaries...")

train_bounds = con.execute(f"""
    SELECT MIN(pickup_hour), MAX(pickup_hour)
    FROM read_parquet('{train_path}')
""").fetchone()

validation_bounds = con.execute(f"""
    SELECT MIN(pickup_hour), MAX(pickup_hour)
    FROM read_parquet('{validation_path}')
""").fetchone()

train_start, train_end = train_bounds
validation_start, validation_end = validation_bounds

print(f"Training period   : {train_start} to {train_end}")
print(f"Validation period : {validation_start} to {validation_end}")

print("\nBuilding direct multi-horizon samples...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE history AS
    SELECT
        pickup_hour,
        CAST(trip_count AS DOUBLE) AS trip_count
    FROM read_parquet('{history_path}')
    ORDER BY pickup_hour
""")

con.execute("""
    CREATE OR REPLACE TEMP TABLE origins AS
    SELECT
        current.pickup_hour AS origin_timestamp,
        current.trip_count AS origin_demand,

        lag24.trip_count AS origin_lag_24,
        lag168.trip_count AS origin_lag_168,

        rolling24.rolling_mean_24,
        rolling168.rolling_mean_168

    FROM history current

    LEFT JOIN history lag24
        ON lag24.pickup_hour =
           current.pickup_hour - INTERVAL 24 HOURS

    LEFT JOIN history lag168
        ON lag168.pickup_hour =
           current.pickup_hour - INTERVAL 168 HOURS

    LEFT JOIN LATERAL (
        SELECT AVG(previous.trip_count) AS rolling_mean_24
        FROM history previous
        WHERE previous.pickup_hour < current.pickup_hour
          AND previous.pickup_hour >=
              current.pickup_hour - INTERVAL 24 HOURS
    ) rolling24 ON TRUE

    LEFT JOIN LATERAL (
        SELECT AVG(previous.trip_count) AS rolling_mean_168
        FROM history previous
        WHERE previous.pickup_hour < current.pickup_hour
          AND previous.pickup_hour >=
              current.pickup_hour - INTERVAL 168 HOURS
    ) rolling168 ON TRUE
""")

con.execute("""
    CREATE OR REPLACE TEMP TABLE candidate_samples AS
    SELECT
        origins.origin_timestamp,
        target.pickup_hour AS target_timestamp,

        EXTRACT(HOUR FROM target.pickup_hour)::INTEGER
            AS target_hour,

        EXTRACT(DOW FROM target.pickup_hour)::INTEGER
            AS target_day_of_week,

        CASE
            WHEN EXTRACT(DOW FROM target.pickup_hour) IN (0, 6)
            THEN 1 ELSE 0
        END AS target_is_weekend,

        EXTRACT(MONTH FROM target.pickup_hour)::INTEGER
            AS target_month,

        horizons.hours_ahead,

        origins.origin_demand,
        origins.origin_lag_24,
        origins.origin_lag_168,
        origins.rolling_mean_24,
        origins.rolling_mean_168,

        target_lag24.trip_count AS target_lag_24,
        target_lag168.trip_count AS target_lag_168,

        target.trip_count AS actual_demand

    FROM origins

    CROSS JOIN range(1, 25) AS horizons(hours_ahead)

    INNER JOIN history target
        ON target.pickup_hour =
           origins.origin_timestamp
           + horizons.hours_ahead * INTERVAL 1 HOUR

    LEFT JOIN history target_lag24
        ON target_lag24.pickup_hour =
           target.pickup_hour - INTERVAL 24 HOURS

    LEFT JOIN history target_lag168
        ON target_lag168.pickup_hour =
           target.pickup_hour - INTERVAL 168 HOURS

    WHERE origins.origin_lag_24 IS NOT NULL
      AND origins.origin_lag_168 IS NOT NULL
      AND origins.rolling_mean_24 IS NOT NULL
      AND origins.rolling_mean_168 IS NOT NULL
      AND target_lag24.trip_count IS NOT NULL
      AND target_lag168.trip_count IS NOT NULL
""")

# Keep only origins with all 24 future target hours available.
con.execute("""
    CREATE OR REPLACE TEMP TABLE complete_origins AS
    SELECT origin_timestamp
    FROM candidate_samples
    GROUP BY origin_timestamp
    HAVING COUNT(*) = 24
""")

# Assign complete forecast windows to chronological splits.
# A window must fit entirely inside its own split.
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE direct_samples AS
    SELECT
        samples.*,

        CASE
            WHEN samples.origin_timestamp >= TIMESTAMP '{train_start}'
             AND samples.origin_timestamp + INTERVAL 24 HOURS
                 <= TIMESTAMP '{train_end}'
            THEN 'train'

            WHEN samples.origin_timestamp >= TIMESTAMP '{validation_start}'
             AND samples.origin_timestamp + INTERVAL 24 HOURS
                 <= TIMESTAMP '{validation_end}'
            THEN 'validation'
        END AS dataset_split

    FROM candidate_samples samples

    INNER JOIN complete_origins complete
        ON complete.origin_timestamp = samples.origin_timestamp

    WHERE
        (
            samples.origin_timestamp >= TIMESTAMP '{train_start}'
            AND samples.origin_timestamp + INTERVAL 24 HOURS
                <= TIMESTAMP '{train_end}'
        )
        OR
        (
            samples.origin_timestamp >= TIMESTAMP '{validation_start}'
            AND samples.origin_timestamp + INTERVAL 24 HOURS
                <= TIMESTAMP '{validation_end}'
        )
""")

con.execute(f"""
    COPY (
        SELECT *
        FROM direct_samples
        ORDER BY origin_timestamp, hours_ahead
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n" + "=" * 70)
print("DIRECT FORECASTING DATASET SUMMARY")
print("=" * 70)

summary = con.execute("""
    SELECT
        dataset_split,
        COUNT(*) AS sample_rows,
        COUNT(DISTINCT origin_timestamp) AS forecast_origins,
        COUNT(DISTINCT hours_ahead) AS horizons,
        MIN(origin_timestamp) AS first_origin,
        MAX(origin_timestamp) AS last_origin
    FROM direct_samples
    GROUP BY dataset_split
    ORDER BY dataset_split
""").fetchdf()

print(summary.to_string(index=False))

print("\nChecking horizon coverage...")

coverage = con.execute("""
    SELECT
        dataset_split,
        MIN(hours_ahead) AS minimum_horizon,
        MAX(hours_ahead) AS maximum_horizon,
        COUNT(DISTINCT hours_ahead) AS unique_horizons
    FROM direct_samples
    GROUP BY dataset_split
    ORDER BY dataset_split
""").fetchdf()

print(coverage.to_string(index=False))

print("\nSample records:")
sample = con.execute("""
    SELECT *
    FROM direct_samples
    ORDER BY origin_timestamp, hours_ahead
    LIMIT 5
""").fetchdf()

print(sample.to_string(index=False))

print(f"\nSaved dataset: {OUTPUT_FILE}")

con.close()
