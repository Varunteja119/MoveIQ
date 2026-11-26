from pathlib import Path
import shutil
import time
import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data/ml/processed"
FEATURES = OUTPUT_DIR / "zone_hourly_demand_features.parquet"
TEMP_DIR = OUTPUT_DIR / "duckdb_temp"

EXPECTED_ZONES = 263
BATCH_SIZE = 100

SPLITS = {
    "train": {
        "start": "2024-01-08 00:00:00",
        "end": "2025-05-28 20:00:00",
        "expected_origins": 2000,
    },
    "validation": {
        "start": "2025-05-28 21:00:00",
        "end": "2025-09-14 09:00:00",
        "expected_origins": 430,
    },
    "test": {
        "start": "2025-09-14 10:00:00",
        "end": "2025-12-31 23:00:00",
        "expected_origins": 430,
    },
}


def log(message):
    print(message, flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{TEMP_DIR.as_posix()}'")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 1000")

    log("=" * 75)
    log("MOVEIQ BATCH-BASED ZONE DIRECT DATASET BUILDER")
    log("=" * 75)

    # Load the feature data once instead of repeatedly reading Parquet.
    log("\nStep 1: Loading zone features into DuckDB...")

    con.execute(f"""
        CREATE TEMP TABLE zone_features AS
        SELECT
            pickup_hour,
            zone_id,
            borough,
            service_zone,
            trip_count,
            hour,
            day_of_week,
            is_weekend,
            month,
            lag_24,
            lag_168,
            rolling_mean_24,
            rolling_mean_168
        FROM read_parquet('{FEATURES.as_posix()}')
    """)

    feature_rows = con.execute(
        "SELECT COUNT(*) FROM zone_features"
    ).fetchone()[0]

    log(f"Loaded feature rows: {feature_rows:,}")

    con.execute("""
        CREATE TEMP TABLE active_zones AS
        SELECT zone_id
        FROM zone_features
        GROUP BY zone_id
        HAVING SUM(trip_count) > 0
    """)

    zone_count = con.execute(
        "SELECT COUNT(*) FROM active_zones"
    ).fetchone()[0]

    if zone_count != EXPECTED_ZONES:
        raise RuntimeError(
            f"Expected {EXPECTED_ZONES} active zones, found {zone_count}"
        )

    representative_zone = con.execute(
        "SELECT MIN(zone_id) FROM active_zones"
    ).fetchone()[0]

    log(f"Active zones: {zone_count}")
    log(f"Representative zone: {representative_zone}")

    con.execute("""
        CREATE TEMP TABLE representative_features AS
        SELECT
            pickup_hour,
            lag_24,
            lag_168,
            rolling_mean_24,
            rolling_mean_168
        FROM zone_features
        WHERE zone_id = ?
    """, [representative_zone])

    con.execute("""
        CREATE TEMP TABLE horizons AS
        SELECT UNNEST(range(1, 25)) AS hours_ahead
    """)

    # Process each split independently.
    for split_name, config in SPLITS.items():
        split_started = time.time()

        start = config["start"]
        end = config["end"]
        expected_origins = config["expected_origins"]

        log("\n" + "=" * 75)
        log(f"STARTING {split_name.upper()} SPLIT")
        log("=" * 75)

        parts_dir = OUTPUT_DIR / f"zone_direct_{split_name}_parts"
        final_path = OUTPUT_DIR / f"zone_direct_{split_name}.parquet"

        if parts_dir.exists():
            shutil.rmtree(parts_dir)

        parts_dir.mkdir(parents=True)

        if final_path.exists():
            final_path.unlink()

        # Identify valid timestamps using one representative zone.
        # The two DST gaps and unavailable target lags are checked here.
        log("Step 2: Identifying valid forecast origins...")

        con.execute("DROP TABLE IF EXISTS valid_origins")

        con.execute(f"""
            CREATE TEMP TABLE valid_origins AS
            WITH candidates AS (
                SELECT
                    o.pickup_hour AS origin_timestamp
                FROM representative_features o
                WHERE
                    o.pickup_hour >= TIMESTAMP '{start}'
                    AND o.pickup_hour + INTERVAL '24 hours'
                        <= TIMESTAMP '{end}'
                    AND DATE_DIFF(
                        'hour',
                        TIMESTAMP '2024-01-01 00:00:00',
                        o.pickup_hour
                    ) % 6 = 0
                    AND o.lag_24 IS NOT NULL
                    AND o.lag_168 IS NOT NULL
                    AND o.rolling_mean_24 IS NOT NULL
                    AND o.rolling_mean_168 IS NOT NULL
            ),
            checked AS (
                SELECT
                    c.origin_timestamp,
                    COUNT(t.pickup_hour) AS target_count,
                    COUNT(t.lag_24) AS valid_lag_24,
                    COUNT(t.lag_168) AS valid_lag_168
                FROM candidates c
                LEFT JOIN representative_features t
                    ON t.pickup_hour > c.origin_timestamp
                    AND t.pickup_hour <=
                        c.origin_timestamp + INTERVAL '24 hours'
                    AND DATE_DIFF(
                        'hour',
                        c.origin_timestamp,
                        t.pickup_hour
                    ) BETWEEN 1 AND 24
                GROUP BY c.origin_timestamp
            )
            SELECT
                origin_timestamp,
                CAST(
                    FLOOR(
                        (ROW_NUMBER() OVER (
                            ORDER BY origin_timestamp
                        ) - 1) / {BATCH_SIZE}.0
                    ) AS INTEGER
                ) AS batch_id
            FROM checked
            WHERE
                target_count = 24
                AND valid_lag_24 = 24
                AND valid_lag_168 = 24
        """)

        origin_count = con.execute(
            "SELECT COUNT(*) FROM valid_origins"
        ).fetchone()[0]

        log(f"Valid forecast origins: {origin_count:,}")

        if origin_count != expected_origins:
            raise RuntimeError(
                f"{split_name}: expected {expected_origins} origins, "
                f"found {origin_count}"
            )

        # Attach each valid origin to all active zones.
        log("Step 3: Preparing zone-specific origins...")

        con.execute("DROP TABLE IF EXISTS split_origins")

        con.execute("""
            CREATE TEMP TABLE split_origins AS
            SELECT
                v.batch_id,
                o.zone_id,
                o.borough,
                o.service_zone,
                o.pickup_hour AS origin_timestamp,
                o.trip_count AS origin_demand,
                o.lag_24 AS origin_lag_24,
                o.lag_168 AS origin_lag_168,
                o.rolling_mean_24,
                o.rolling_mean_168
            FROM valid_origins v
            INNER JOIN zone_features o
                ON o.pickup_hour = v.origin_timestamp
            INNER JOIN active_zones z
                ON z.zone_id = o.zone_id
            WHERE
                o.lag_24 IS NOT NULL
                AND o.lag_168 IS NOT NULL
                AND o.rolling_mean_24 IS NOT NULL
                AND o.rolling_mean_168 IS NOT NULL
        """)

        zone_origins = con.execute(
            "SELECT COUNT(*) FROM split_origins"
        ).fetchone()[0]

        represented_zones = con.execute(
            "SELECT COUNT(DISTINCT zone_id) FROM split_origins"
        ).fetchone()[0]

        expected_zone_origins = expected_origins * EXPECTED_ZONES

        if zone_origins != expected_zone_origins:
            raise RuntimeError(
                f"{split_name}: expected {expected_zone_origins:,} "
                f"zone origins, found {zone_origins:,}"
            )

        if represented_zones != EXPECTED_ZONES:
            raise RuntimeError(
                f"{split_name}: expected {EXPECTED_ZONES} zones, "
                f"found {represented_zones}"
            )

        log(f"Zone origins: {zone_origins:,}")
        log(f"Zones represented: {represented_zones}")

        # Build manageable batches of 100 forecast timestamps.
        batch_ids = [
            row[0] for row in con.execute("""
                SELECT DISTINCT batch_id
                FROM valid_origins
                ORDER BY batch_id
            """).fetchall()
        ]

        log(f"Step 4: Writing {len(batch_ids)} batches...")

        for index, batch_id in enumerate(batch_ids, start=1):
            batch_path = parts_dir / f"part_{index:04d}.parquet"

            con.execute(f"""
                COPY (
                    SELECT
                        o.zone_id,
                        o.borough,
                        o.service_zone,

                        o.origin_timestamp,
                        t.pickup_hour AS target_timestamp,

                        t.hour AS target_hour,
                        t.day_of_week AS target_day_of_week,
                        t.is_weekend AS target_is_weekend,
                        t.month AS target_month,

                        h.hours_ahead,

                        o.origin_demand,
                        o.origin_lag_24,
                        o.origin_lag_168,
                        o.rolling_mean_24,
                        o.rolling_mean_168,

                        t.lag_24 AS target_lag_24,
                        t.lag_168 AS target_lag_168,

                        t.trip_count AS actual_demand,

                        '{split_name}' AS dataset_split

                    FROM split_origins o
                    CROSS JOIN horizons h
                    INNER JOIN zone_features t
                        ON t.zone_id = o.zone_id
                        AND t.pickup_hour =
                            o.origin_timestamp
                            + h.hours_ahead * INTERVAL '1 hour'

                    WHERE o.batch_id = {batch_id}

                ) TO '{batch_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            if index == 1 or index % 2 == 0 or index == len(batch_ids):
                log(
                    f"Batch {index}/{len(batch_ids)} completed "
                    f"({index * BATCH_SIZE:,} forecast origins attempted)"
                )

        # Combine the batch files into one Parquet dataset.
        log("Step 5: Combining batch files...")

        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet('{(parts_dir / "*.parquet").as_posix()}')
            )
            TO '{final_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # Verify the final file.
        log("Step 6: Verifying final dataset...")

        result = con.execute(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT zone_id) AS zones,
                COUNT(DISTINCT origin_timestamp) AS origins,
                COUNT(*) FILTER (
                    WHERE target_lag_24 IS NULL
                       OR target_lag_168 IS NULL
                       OR actual_demand IS NULL
                       OR actual_demand < 0
                ) AS invalid_rows
            FROM read_parquet('{final_path.as_posix()}')
        """).fetchone()

        rows, zones, origins, invalid_rows = result
        expected_rows = expected_zone_origins * 24

        log(f"Final rows: {rows:,}")
        log(f"Zones: {zones}")
        log(f"Forecast origins: {origins:,}")
        log(f"Invalid rows: {invalid_rows}")

        if rows != expected_rows:
            raise RuntimeError(
                f"{split_name}: expected {expected_rows:,} rows, "
                f"found {rows:,}"
            )

        if zones != EXPECTED_ZONES or invalid_rows != 0:
            raise RuntimeError(f"{split_name}: validation failed")

        size_mb = final_path.stat().st_size / (1024 * 1024)
        elapsed = (time.time() - split_started) / 60

        log(f"Saved: {final_path}")
        log(f"File size: {size_mb:,.2f} MB")
        log(f"Elapsed time: {elapsed:.2f} minutes")

        # Remove temporary batch files only after successful validation.
        shutil.rmtree(parts_dir)

        con.execute("DROP TABLE split_origins")
        con.execute("DROP TABLE valid_origins")

    con.close()

    log("\n" + "=" * 75)
    log("ALL THREE ZONE DIRECT DATASETS COMPLETED")
    log("=" * 75)


if __name__ == "__main__":
    main()
