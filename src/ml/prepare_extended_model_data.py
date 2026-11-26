from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "ml" / "processed"

CITY_FEATURES = PROCESSED / "hourly_demand_features_2024_2026.parquet"
ZONE_FEATURES = PROCESSED / "zone_hourly_demand_features_2024_2026.parquet"

CITY_SPLITS = {
    "train": {
        "start": "2024-01-01 00:00:00",
        "end": "2026-01-01 00:00:00",
        "output": "train_2024_2026.parquet",
    },
    "validation": {
        "start": "2026-01-01 00:00:00",
        "end": "2026-04-01 00:00:00",
        "output": "validation_2024_2026.parquet",
    },
    "test": {
        "start": "2026-04-01 00:00:00",
        "end": "2026-10-01 00:00:00",
        "output": "test_2024_2026.parquet",
    },
}

ZONE_SPLITS = {
    "train": {
        "start": "2024-01-01 00:00:00",
        "end": "2026-01-01 00:00:00",
        "output": "zone_train_2024_2026.parquet",
    },
    "validation": {
        "start": "2026-01-01 00:00:00",
        "end": "2026-04-01 00:00:00",
        "output": "zone_validation_2024_2026.parquet",
    },
    "test": {
        "start": "2026-04-01 00:00:00",
        "end": "2026-10-01 00:00:00",
        "output": "zone_test_2024_2026.parquet",
    },
}

REQUIRED_FEATURES = """
    lag_1 IS NOT NULL
    AND lag_24 IS NOT NULL
    AND lag_168 IS NOT NULL
    AND rolling_mean_24 IS NOT NULL
    AND rolling_mean_168 IS NOT NULL
"""


def log(message):
    print(message, flush=True)


def validate_file(con, path, expected_zones=None):
    source = path.resolve().as_posix()

    if expected_zones is None:
        result = con.execute(f"""
            SELECT
                COUNT(*) AS records,
                COUNT(DISTINCT pickup_hour) AS hours,
                MIN(pickup_hour) AS first_hour,
                MAX(pickup_hour) AS last_hour,
                SUM(trip_count) AS total_pickups
            FROM read_parquet('{source}')
        """).fetchone()

        records, hours, first_hour, last_hour, pickups = result
        zones = None

    else:
        result = con.execute(f"""
            SELECT
                COUNT(*) AS records,
                COUNT(DISTINCT pickup_hour) AS hours,
                MIN(pickup_hour) AS first_hour,
                MAX(pickup_hour) AS last_hour,
                SUM(trip_count) AS total_pickups,
                COUNT(DISTINCT zone_id) AS zones
            FROM read_parquet('{source}')
        """).fetchone()

        records, hours, first_hour, last_hour, pickups, zones = result

    if records == 0:
        raise RuntimeError(f"Empty output dataset: {path.name}")

    if expected_zones is not None and zones != expected_zones:
        raise RuntimeError(
            f"{path.name}: expected {expected_zones} zones, found {zones}"
        )

    return {
        "records": records,
        "hours": hours,
        "first_hour": first_hour,
        "last_hour": last_hour,
        "pickups": pickups,
        "zones": zones,
    }


def main():
    log("=" * 78)
    log("MOVEIQ — EXTENDED CHRONOLOGICAL DATA SPLITS")
    log("=" * 78)

    for path in [CITY_FEATURES, ZONE_FEATURES]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input dataset: {path}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")

    try:
        # Identify the active zones across the complete extended history.
        log("\nIdentifying active taxi zones...")

        zone_source = ZONE_FEATURES.resolve().as_posix()

        con.execute(f"""
            CREATE TEMP TABLE active_zones AS
            SELECT zone_id
            FROM read_parquet('{zone_source}')
            GROUP BY zone_id
            HAVING SUM(trip_count) > 0
        """)

        active_count = con.execute(
            "SELECT COUNT(*) FROM active_zones"
        ).fetchone()[0]

        log(f"Active zones found: {active_count}")

        if active_count != 263:
            raise RuntimeError(
                f"Expected 263 active zones, found {active_count}"
            )

        # Build citywide datasets.
        log("\n" + "=" * 78)
        log("CITYWIDE DATASETS")
        log("=" * 78)

        city_source = CITY_FEATURES.resolve().as_posix()
        city_results = {}

        for split_name, config in CITY_SPLITS.items():
            output = PROCESSED / config["output"]
            output_path = output.resolve().as_posix()

            log(f"\nCreating citywide {split_name} dataset...")

            con.execute(f"""
                COPY (
                    SELECT *
                    FROM read_parquet('{city_source}')
                    WHERE
                        ({REQUIRED_FEATURES})
                        AND pickup_hour >= TIMESTAMP '{config["start"]}'
                        AND pickup_hour < TIMESTAMP '{config["end"]}'
                    ORDER BY pickup_hour
                )
                TO '{output_path}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            stats = validate_file(con, output)
            city_results[split_name] = stats

            log(
                f"Records: {stats['records']:,} | "
                f"From: {stats['first_hour']} | "
                f"To: {stats['last_hour']}"
            )
            log(f"Saved: {output}")

        # Build zone-level datasets.
        log("\n" + "=" * 78)
        log("ZONE-LEVEL DATASETS")
        log("=" * 78)

        zone_results = {}

        for split_name, config in ZONE_SPLITS.items():
            output = PROCESSED / config["output"]
            output_path = output.resolve().as_posix()

            log(f"\nCreating zone {split_name} dataset...")

            con.execute(f"""
                COPY (
                    SELECT f.*
                    FROM read_parquet('{zone_source}') f
                    INNER JOIN active_zones z USING (zone_id)
                    WHERE
                        ({REQUIRED_FEATURES})
                        AND f.pickup_hour >= TIMESTAMP '{config["start"]}'
                        AND f.pickup_hour < TIMESTAMP '{config["end"]}'
                    ORDER BY f.pickup_hour, f.zone_id
                )
                TO '{output_path}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            stats = validate_file(
                con,
                output,
                expected_zones=active_count,
            )
            zone_results[split_name] = stats

            log(
                f"Records: {stats['records']:,} | "
                f"Zones: {stats['zones']} | "
                f"Hours: {stats['hours']:,}"
            )
            log(
                f"From: {stats['first_hour']} | "
                f"To: {stats['last_hour']}"
            )
            log(f"Saved: {output}")

        # Verify split totals reconcile with the complete usable datasets.
        log("\n" + "=" * 78)
        log("FINAL RECONCILIATION")
        log("=" * 78)

        for label, results, source_path, is_zone in [
            ("CITYWIDE", city_results, CITY_FEATURES, False),
            ("ZONE", zone_results, ZONE_FEATURES, True),
        ]:
            split_records = sum(x["records"] for x in results.values())
            split_pickups = sum(x["pickups"] for x in results.values())

            source = source_path.resolve().as_posix()

            if is_zone:
                source_stats = con.execute(f"""
                    SELECT COUNT(*), SUM(trip_count)
                    FROM read_parquet('{source}') f
                    INNER JOIN active_zones z USING (zone_id)
                    WHERE {REQUIRED_FEATURES}
                """).fetchone()
            else:
                source_stats = con.execute(f"""
                    SELECT COUNT(*), SUM(trip_count)
                    FROM read_parquet('{source}')
                    WHERE {REQUIRED_FEATURES}
                """).fetchone()

            source_records, source_pickups = source_stats

            log(f"\n{label}")
            log(f"Source usable records : {source_records:,}")
            log(f"Split records         : {split_records:,}")
            log(f"Source pickups        : {source_pickups:,.0f}")
            log(f"Split pickups         : {split_pickups:,.0f}")

            if split_records != source_records:
                raise RuntimeError(
                    f"{label}: split records do not reconcile."
                )

            if abs(float(split_pickups) - float(source_pickups)) > 0.01:
                raise RuntimeError(
                    f"{label}: pickup totals do not reconcile."
                )

        log("\n" + "=" * 78)
        log("SUCCESS: ALL EXTENDED CHRONOLOGICAL SPLITS CREATED")
        log("=" * 78)
        log("Original 2024–2025 datasets and models were not modified.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
