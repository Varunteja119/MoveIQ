from pathlib import Path
import shutil
import time
import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml" / "processed"
TEMP = DATA / "duckdb_temp_extended_direct"

CITY_FEATURES = DATA / "hourly_demand_features_2024_2026.parquet"
ZONE_FEATURES = DATA / "zone_hourly_demand_features_2024_2026.parquet"

EXPECTED_ZONES = 263
BATCH_SIZE = 100

SPLITS = {
    "train": {
        "start": "2024-01-01 00:00:00",
        "end": "2026-01-01 00:00:00",
    },
    "validation": {
        "start": "2026-01-01 00:00:00",
        "end": "2026-04-01 00:00:00",
    },
    "test": {
        "start": "2026-04-01 00:00:00",
        "end": "2026-10-01 00:00:00",
    },
}


def log(message):
    print(message, flush=True)


def check_inputs():
    for path in [CITY_FEATURES, ZONE_FEATURES]:
        if not path.exists():
            raise FileNotFoundError(path)


def setup_connection():
    TEMP.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{TEMP.as_posix()}'")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 1000")

    return con


def build_citywide(con):
    log("\n" + "=" * 78)
    log("CITYWIDE DIRECT 24-HOUR DATASETS")
    log("=" * 78)

    source = CITY_FEATURES.resolve().as_posix()

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE city_features AS
        SELECT
            pickup_hour,
            CAST(trip_count AS DOUBLE) AS trip_count,
            hour,
            day_of_week,
            is_weekend,
            month,
            lag_24,
            lag_168,
            rolling_mean_24,
            rolling_mean_168
        FROM read_parquet('{source}')
    """)

    for split_name, config in SPLITS.items():
        started = time.time()

        start = config["start"]
        end = config["end"]

        output = DATA / f"direct_{split_name}_2024_2026.parquet"
        output_path = output.resolve().as_posix()

        log(f"\nBuilding citywide {split_name}...")

        con.execute("DROP TABLE IF EXISTS city_origins")
        con.execute("DROP TABLE IF EXISTS city_candidates")
        con.execute("DROP TABLE IF EXISTS city_complete_origins")

        # Every origin must have complete historical features.
        con.execute(f"""
            CREATE TEMP TABLE city_origins AS
            SELECT
                pickup_hour AS origin_timestamp,
                trip_count AS origin_demand,
                lag_24 AS origin_lag_24,
                lag_168 AS origin_lag_168,
                rolling_mean_24,
                rolling_mean_168
            FROM city_features
            WHERE
                pickup_hour >= TIMESTAMP '{start}'
                AND pickup_hour + INTERVAL 24 HOURS
                    < TIMESTAMP '{end}'
                AND lag_24 IS NOT NULL
                AND lag_168 IS NOT NULL
                AND rolling_mean_24 IS NOT NULL
                AND rolling_mean_168 IS NOT NULL
        """)

        # Generate the 24 future target rows for every origin.
        # Target lags must also be available at prediction time.
        con.execute("""
            CREATE TEMP TABLE city_candidates AS
            SELECT
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

                t.trip_count AS actual_demand

            FROM city_origins o

            CROSS JOIN range(1, 25) AS h(hours_ahead)

            INNER JOIN city_features t
                ON t.pickup_hour =
                   o.origin_timestamp
                   + h.hours_ahead * INTERVAL 1 HOUR

            WHERE
                t.lag_24 IS NOT NULL
                AND t.lag_168 IS NOT NULL
        """)

        con.execute("""
            CREATE TEMP TABLE city_complete_origins AS
            SELECT origin_timestamp
            FROM city_candidates
            GROUP BY origin_timestamp
            HAVING COUNT(*) = 24
        """)

        origin_count = con.execute("""
            SELECT COUNT(*) FROM city_complete_origins
        """).fetchone()[0]

        expected_rows = origin_count * 24

        log(f"Complete forecast origins: {origin_count:,}")
        log(f"Expected sample rows: {expected_rows:,}")

        if origin_count == 0:
            raise RuntimeError(
                f"Citywide {split_name}: no complete forecast origins."
            )

        con.execute(f"""
            COPY (
                SELECT
                    c.*,
                    '{split_name}' AS dataset_split
                FROM city_candidates c
                INNER JOIN city_complete_origins o
                    USING (origin_timestamp)
                ORDER BY origin_timestamp, hours_ahead
            )
            TO '{output_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        result = con.execute(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT origin_timestamp) AS origins,
                COUNT(DISTINCT hours_ahead) AS horizons,
                MIN(origin_timestamp) AS first_origin,
                MAX(origin_timestamp) AS last_origin,
                MIN(target_timestamp) AS first_target,
                MAX(target_timestamp) AS last_target,
                COUNT(*) FILTER (
                    WHERE actual_demand IS NULL
                       OR target_lag_24 IS NULL
                       OR target_lag_168 IS NULL
                       OR actual_demand < 0
                ) AS invalid_rows,
                MIN(hours_ahead) AS min_horizon,
                MAX(hours_ahead) AS max_horizon
            FROM read_parquet('{output_path}')
        """).fetchone()

        (
            rows, origins, horizons, first_origin, last_origin,
            first_target, last_target, invalid_rows,
            min_horizon, max_horizon
        ) = result

        log(f"Rows: {rows:,}")
        log(f"Origins: {origins:,}")
        log(f"Horizons: {horizons} ({min_horizon}–{max_horizon})")
        log(f"Origin range: {first_origin} to {last_origin}")
        log(f"Target range: {first_target} to {last_target}")
        log(f"Invalid rows: {invalid_rows}")

        if rows != expected_rows:
            raise RuntimeError(
                f"Citywide {split_name}: expected {expected_rows:,} "
                f"rows, found {rows:,}"
            )

        if horizons != 24 or min_horizon != 1 or max_horizon != 24:
            raise RuntimeError(
                f"Citywide {split_name}: horizon coverage failed."
            )

        if invalid_rows != 0:
            raise RuntimeError(
                f"Citywide {split_name}: invalid rows found."
            )

        size_mb = output.stat().st_size / (1024 * 1024)

        log(f"Saved: {output}")
        log(f"File size: {size_mb:.2f} MB")
        log(f"Elapsed: {(time.time() - started) / 60:.2f} minutes")


def build_zone_level(con):
    log("\n" + "=" * 78)
    log("ZONE-LEVEL DIRECT 24-HOUR DATASETS")
    log("=" * 78)

    source = ZONE_FEATURES.resolve().as_posix()

    log("\nLoading zone features into DuckDB...")

    con.execute(f"""
        CREATE TEMP TABLE zone_features AS
        SELECT
            pickup_hour,
            zone_id,
            borough,
            service_zone,
            CAST(trip_count AS DOUBLE) AS trip_count,
            hour,
            day_of_week,
            is_weekend,
            month,
            lag_24,
            lag_168,
            rolling_mean_24,
            rolling_mean_168
        FROM read_parquet('{source}')
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

    for split_name, config in SPLITS.items():
        started = time.time()

        start = config["start"]
        end = config["end"]

        parts_dir = DATA / f"direct_zone_{split_name}_2024_2026_parts"
        final_path = DATA / f"direct_zone_{split_name}_2024_2026.parquet"

        log("\n" + "-" * 78)
        log(f"ZONE {split_name.upper()}")
        log("-" * 78)

        # Recreate only this split's temporary batch directory.
        if parts_dir.exists():
            shutil.rmtree(parts_dir)

        parts_dir.mkdir(parents=True, exist_ok=True)

        if final_path.exists():
            final_path.unlink()

        con.execute("DROP TABLE IF EXISTS zone_valid_origins")

        # Sample every six hours, anchored at 2024-01-01.
        # Each origin must have all 24 future hours and their target lags.
        con.execute(f"""
            CREATE TEMP TABLE zone_valid_origins AS
            WITH candidates AS (
                SELECT
                    o.pickup_hour AS origin_timestamp
                FROM representative_features o
                WHERE
                    o.pickup_hour >= TIMESTAMP '{start}'
                    AND o.pickup_hour + INTERVAL 24 HOURS
                        < TIMESTAMP '{end}'
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
                        c.origin_timestamp + INTERVAL 24 HOURS
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
                        ) - 1
                    ) / {BATCH_SIZE}.0)
                    AS INTEGER
                ) AS batch_id
            FROM checked
            WHERE
                target_count = 24
                AND valid_lag_24 = 24
                AND valid_lag_168 = 24
        """)

        origin_count = con.execute(
            "SELECT COUNT(*) FROM zone_valid_origins"
        ).fetchone()[0]

        log(f"Valid six-hour forecast origins: {origin_count:,}")

        if origin_count == 0:
            raise RuntimeError(
                f"Zone {split_name}: no valid forecast origins."
            )

        expected_zone_origins = origin_count * EXPECTED_ZONES

        con.execute("DROP TABLE IF EXISTS zone_split_origins")

        con.execute("""
            CREATE TEMP TABLE zone_split_origins AS
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
            FROM zone_valid_origins v
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

        actual_zone_origins = con.execute(
            "SELECT COUNT(*) FROM zone_split_origins"
        ).fetchone()[0]

        represented_zones = con.execute(
            "SELECT COUNT(DISTINCT zone_id) FROM zone_split_origins"
        ).fetchone()[0]

        if actual_zone_origins != expected_zone_origins:
            raise RuntimeError(
                f"{split_name}: expected {expected_zone_origins:,} "
                f"zone origins, found {actual_zone_origins:,}"
            )

        if represented_zones != EXPECTED_ZONES:
            raise RuntimeError(
                f"{split_name}: expected {EXPECTED_ZONES} zones, "
                f"found {represented_zones}"
            )

        batch_ids = [
            row[0]
            for row in con.execute("""
                SELECT DISTINCT batch_id
                FROM zone_valid_origins
                ORDER BY batch_id
            """).fetchall()
        ]

        log(f"Zone-specific origins: {actual_zone_origins:,}")
        log(f"Batch count: {len(batch_ids)}")

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

                    FROM zone_split_origins o

                    CROSS JOIN horizons h

                    INNER JOIN zone_features t
                        ON t.zone_id = o.zone_id
                        AND t.pickup_hour =
                            o.origin_timestamp
                            + h.hours_ahead * INTERVAL 1 HOUR

                    WHERE o.batch_id = {batch_id}
                )
                TO '{batch_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            if (
                index == 1
                or index % 2 == 0
                or index == len(batch_ids)
            ):
                log(f"Completed batch {index}/{len(batch_ids)}")

        log("Combining zone batches...")

        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet(
                    '{(parts_dir / "*.parquet").as_posix()}'
                )
            )
            TO '{final_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        result = con.execute(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT zone_id) AS zones,
                COUNT(DISTINCT origin_timestamp) AS origins,
                COUNT(DISTINCT hours_ahead) AS horizons,
                MIN(hours_ahead) AS min_horizon,
                MAX(hours_ahead) AS max_horizon,
                MIN(origin_timestamp) AS first_origin,
                MAX(origin_timestamp) AS last_origin,
                MIN(target_timestamp) AS first_target,
                MAX(target_timestamp) AS last_target,
                COUNT(*) FILTER (
                    WHERE actual_demand IS NULL
                       OR target_lag_24 IS NULL
                       OR target_lag_168 IS NULL
                       OR actual_demand < 0
                ) AS invalid_rows
            FROM read_parquet('{final_path.as_posix()}')
        """).fetchone()

        (
            rows, zones, origins, horizons_count,
            min_horizon, max_horizon,
            first_origin, last_origin,
            first_target, last_target,
            invalid_rows
        ) = result

        expected_rows = expected_zone_origins * 24

        log(f"Rows: {rows:,}")
        log(f"Zones: {zones}")
        log(f"Forecast origins: {origins:,}")
        log(f"Horizons: {horizons_count} ({min_horizon}–{max_horizon})")
        log(f"Origin range: {first_origin} to {last_origin}")
        log(f"Target range: {first_target} to {last_target}")
        log(f"Invalid rows: {invalid_rows}")

        if rows != expected_rows:
            raise RuntimeError(
                f"{split_name}: expected {expected_rows:,} rows, "
                f"found {rows:,}"
            )

        if zones != EXPECTED_ZONES:
            raise RuntimeError(f"{split_name}: zone coverage failed.")

        if origins != origin_count:
            raise RuntimeError(f"{split_name}: origin coverage failed.")

        if horizons_count != 24 or min_horizon != 1 or max_horizon != 24:
            raise RuntimeError(f"{split_name}: horizon coverage failed.")

        if invalid_rows != 0:
            raise RuntimeError(f"{split_name}: invalid rows found.")

        size_mb = final_path.stat().st_size / (1024 * 1024)

        log(f"Saved: {final_path}")
        log(f"File size: {size_mb:.2f} MB")
        log(f"Elapsed: {(time.time() - started) / 60:.2f} minutes")

        # Keep final output; remove temporary batches only after validation.
        shutil.rmtree(parts_dir)

        con.execute("DROP TABLE zone_split_origins")
        con.execute("DROP TABLE zone_valid_origins")


def main():
    check_inputs()

    log("=" * 78)
    log("MOVEIQ — EXTENDED DIRECT FORECASTING DATASET BUILDER")
    log("=" * 78)

    con = setup_connection()

    try:
        build_citywide(con)
        build_zone_level(con)

        log("\n" + "=" * 78)
        log("SUCCESS: ALL SIX DIRECT DATASETS CREATED AND VALIDATED")
        log("=" * 78)
        log("Original 2024–2025 datasets and models were not modified.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
