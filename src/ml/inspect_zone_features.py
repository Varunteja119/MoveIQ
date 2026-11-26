from pathlib import Path
import duckdb

ZONE_FILE = Path("data/ml/processed/zone_hourly_demand_2024_2025.parquet")
LOOKUP_FILE = Path("data/reference/taxi_zone_lookup.csv")

con = duckdb.connect()

zone_path = ZONE_FILE.resolve().as_posix()
lookup_path = LOOKUP_FILE.resolve().as_posix()

print("\n" + "=" * 75)
print("TAXI ZONE LOOKUP STRUCTURE")
print("=" * 75)

print(con.execute(f"""
    DESCRIBE SELECT * FROM read_csv_auto('{lookup_path}')
""").fetchdf().to_string(index=False))

print("\n" + "=" * 75)
print("BOROUGH DISTRIBUTION")
print("=" * 75)

print(con.execute(f"""
    SELECT
        l.Borough,
        COUNT(DISTINCT l.LocationID) AS zones,
        SUM(z.trip_count) AS total_pickups,
        ROUND(AVG(z.trip_count), 2) AS avg_hourly_zone_demand
    FROM read_csv_auto('{lookup_path}') l
    JOIN read_parquet('{zone_path}') z
        ON l.LocationID = z.zone_id
    GROUP BY l.Borough
    ORDER BY total_pickups DESC
""").fetchdf().to_string(index=False))

print("\n" + "=" * 75)
print("ZONES WITH ZERO HISTORICAL PICKUPS")
print("=" * 75)

print(con.execute(f"""
    SELECT
        l.LocationID,
        l.Borough,
        l.Zone,
        SUM(z.trip_count) AS total_pickups
    FROM read_csv_auto('{lookup_path}') l
    JOIN read_parquet('{zone_path}') z
        ON l.LocationID = z.zone_id
    GROUP BY l.LocationID, l.Borough, l.Zone
    HAVING SUM(z.trip_count) = 0
    ORDER BY l.LocationID
""").fetchdf().to_string(index=False))

print("\n" + "=" * 75)
print("ZONE DATA TIMESTAMP CONTINUITY")
print("=" * 75)

print(con.execute(f"""
    WITH hours AS (
        SELECT DISTINCT pickup_hour
        FROM read_parquet('{zone_path}')
    ),
    ordered AS (
        SELECT
            pickup_hour,
            LAG(pickup_hour) OVER (ORDER BY pickup_hour) AS previous_hour
        FROM hours
    )
    SELECT
        COUNT(*) AS unique_hours,
        SUM(
            CASE WHEN previous_hour IS NOT NULL
                AND pickup_hour - previous_hour != INTERVAL '1 hour'
            THEN 1 ELSE 0 END
        ) AS non_consecutive_transitions
    FROM ordered
""").fetchdf().to_string(index=False))

print("\n" + "=" * 75)
print("ZONE DATA SAMPLE")
print("=" * 75)

print(con.execute(f"""
    SELECT *
    FROM read_parquet('{zone_path}')
    ORDER BY pickup_hour, zone_id
    LIMIT 10
""").fetchdf().to_string(index=False))

con.close()
