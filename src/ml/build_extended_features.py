from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ml/processed"
LOOKUP = ROOT / "data/reference/taxi_zone_lookup.csv"

CITY_INPUT = DATA / "hourly_demand_2024_2026.parquet"
ZONE_INPUT = DATA / "zone_hourly_demand_2024_2026.parquet"

CITY_OUTPUT = DATA / "hourly_demand_features_2024_2026.parquet"
ZONE_OUTPUT = DATA / "zone_hourly_demand_features_2024_2026.parquet"


def build_city_features(con):
    source = CITY_INPUT.resolve().as_posix()
    destination = CITY_OUTPUT.resolve().as_posix()

    print("\nBuilding citywide features...")

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
                    ON lag1.pickup_hour =
                       b.pickup_hour - INTERVAL 1 HOUR
                LEFT JOIN base lag24
                    ON lag24.pickup_hour =
                       b.pickup_hour - INTERVAL 24 HOURS
                LEFT JOIN base lag168
                    ON lag168.pickup_hour =
                       b.pickup_hour - INTERVAL 168 HOURS
                LEFT JOIN LATERAL (
                    SELECT AVG(p.trip_count) AS rolling_mean_24
                    FROM base p
                    WHERE p.pickup_hour < b.pickup_hour
                      AND p.pickup_hour >=
                          b.pickup_hour - INTERVAL 24 HOURS
                ) rolling24 ON TRUE
                LEFT JOIN LATERAL (
                    SELECT AVG(p.trip_count) AS rolling_mean_168
                    FROM base p
                    WHERE p.pickup_hour < b.pickup_hour
                      AND p.pickup_hour >=
                          b.pickup_hour - INTERVAL 168 HOURS
                ) rolling168 ON TRUE
            )
            SELECT *
            FROM features
            ORDER BY pickup_hour
        )
        TO '{destination}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    result = con.execute(f"""
        SELECT
            COUNT(*) AS records,
            COUNT(lag_1) AS valid_lag_1,
            COUNT(lag_24) AS valid_lag_24,
            COUNT(lag_168) AS valid_lag_168,
            COUNT(rolling_mean_24) AS valid_rolling_24,
            COUNT(rolling_mean_168) AS valid_rolling_168,
            MIN(pickup_hour) AS first_hour,
            MAX(pickup_hour) AS last_hour
        FROM read_parquet('{destination}')
    """).fetchdf()

    print(result.to_string(index=False))
    print(f"Saved: {CITY_OUTPUT}")


def build_zone_features(con):
    source = ZONE_INPUT.resolve().as_posix()
    lookup = LOOKUP.resolve().as_posix()
    destination = ZONE_OUTPUT.resolve().as_posix()

    print("\nBuilding zone-level features...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE zone_base AS
        SELECT
            d.pickup_hour,
            d.zone_id,
            d.trip_count,
            COALESCE(l.Borough, 'Unknown') AS borough,
            COALESCE(l.Zone, 'Unknown') AS zone,
            COALESCE(l.service_zone, 'Unknown') AS service_zone,
            EXTRACT(HOUR FROM d.pickup_hour)::INTEGER AS hour,
            ((DAYOFWEEK(d.pickup_hour) + 5) % 7)::INTEGER
                AS day_of_week,
            CASE
                WHEN DAYOFWEEK(d.pickup_hour) IN (0, 6)
                THEN 1 ELSE 0
            END AS is_weekend,
            EXTRACT(MONTH FROM d.pickup_hour)::INTEGER AS month
        FROM read_parquet('{source}') d
        LEFT JOIN read_csv_auto('{lookup}') l
            ON d.zone_id = l.LocationID
    """)

    print("Calculating exact previous-hour lag...")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE zone_lag_features AS
        SELECT
            current.*,
            previous_1.trip_count AS lag_1
        FROM zone_base current
        LEFT JOIN zone_base previous_1
            ON current.zone_id = previous_1.zone_id
            AND previous_1.pickup_hour =
                current.pickup_hour - INTERVAL '1 hour'
    """)

    print("Calculating exact historical lags...")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE zone_exact_lags AS
        SELECT
            current.*,
            previous_24.trip_count AS lag_24,
            previous_168.trip_count AS lag_168
        FROM zone_lag_features current
        LEFT JOIN zone_base previous_24
            ON current.zone_id = previous_24.zone_id
            AND previous_24.pickup_hour =
                current.pickup_hour - INTERVAL '24 hours'
        LEFT JOIN zone_base previous_168
            ON current.zone_id = previous_168.zone_id
            AND previous_168.pickup_hour =
                current.pickup_hour - INTERVAL '168 hours'
    """)

    print("Calculating rolling averages and writing output...")

    con.execute(f"""
        COPY (
            SELECT
                *,
                AVG(trip_count) OVER (
                    PARTITION BY zone_id
                    ORDER BY pickup_hour
                    RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                    AND INTERVAL '1 hour' PRECEDING
                ) AS rolling_mean_24,
                AVG(trip_count) OVER (
                    PARTITION BY zone_id
                    ORDER BY pickup_hour
                    RANGE BETWEEN INTERVAL '168 hours' PRECEDING
                    AND INTERVAL '1 hour' PRECEDING
                ) AS rolling_mean_168
            FROM zone_exact_lags
            ORDER BY pickup_hour, zone_id
        )
        TO '{destination}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    result = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT zone_id) AS zones,
            COUNT(DISTINCT pickup_hour) AS hours,
            SUM(trip_count) AS total_pickups,
            COUNT(*) FILTER (WHERE lag_1 IS NULL) AS null_lag_1,
            COUNT(*) FILTER (WHERE lag_24 IS NULL) AS null_lag_24,
            COUNT(*) FILTER (WHERE lag_168 IS NULL) AS null_lag_168,
            COUNT(*) FILTER (
                WHERE rolling_mean_24 IS NULL
            ) AS null_rolling_24,
            COUNT(*) FILTER (
                WHERE rolling_mean_168 IS NULL
            ) AS null_rolling_168
        FROM read_parquet('{destination}')
    """).fetchdf()

    print(result.to_string(index=False))
    print(f"Saved: {ZONE_OUTPUT}")


def main():
    for path in [CITY_INPUT, ZONE_INPUT, LOOKUP]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 75)
    print("MOVEIQ — EXTENDED FEATURE ENGINEERING")
    print("=" * 75)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")

    build_city_features(con)
    build_zone_features(con)

    print("\nFeature engineering completed successfully.")
    con.close()


if __name__ == "__main__":
    main()