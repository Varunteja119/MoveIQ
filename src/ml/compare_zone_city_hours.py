from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "ml" / "processed"

CITY_FILE = (PROCESSED / "hourly_demand_2024_2025.parquet").as_posix()
ZONE_FILE = (PROCESSED / "zone_hourly_demand_2024_2025.parquet").as_posix()

con = duckdb.connect()

print("=" * 75)
print("HOURLY PICKUP RECONCILIATION")
print("=" * 75)

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW city AS
    SELECT
        pickup_hour,
        trip_count AS city_pickups
    FROM read_parquet('{CITY_FILE}')
""")

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW zone AS
    SELECT
        pickup_hour,
        SUM(trip_count) AS zone_pickups
    FROM read_parquet('{ZONE_FILE}')
    GROUP BY pickup_hour
""")

comparison = con.execute("""
    SELECT
        city.pickup_hour,
        city.city_pickups,
        zone.zone_pickups,
        city.city_pickups - zone.zone_pickups AS difference
    FROM city
    INNER JOIN zone USING (pickup_hour)
    WHERE city.city_pickups != zone.zone_pickups
    ORDER BY ABS(city.city_pickups - zone.zone_pickups) DESC
""").fetchdf()

print(f"\nHours with different pickup counts: {len(comparison):,}")

if not comparison.empty:
    print("\nLargest hourly differences:")
    print(comparison.head(30).to_string(index=False))

summary = con.execute("""
    SELECT
        SUM(city.city_pickups) AS city_total,
        SUM(zone.zone_pickups) AS zone_total,
        SUM(city.city_pickups - zone.zone_pickups) AS net_difference,
        COUNT(*) FILTER (
            WHERE city.city_pickups > zone.zone_pickups
        ) AS hours_city_higher,
        COUNT(*) FILTER (
            WHERE city.city_pickups < zone.zone_pickups
        ) AS hours_zone_higher
    FROM city
    INNER JOIN zone USING (pickup_hour)
""").fetchone()

print("\nSUMMARY")
print(f"Citywide total       : {summary[0]:,.0f}")
print(f"Zone total           : {summary[1]:,.0f}")
print(f"Net difference       : {summary[2]:,.0f}")
print(f"Hours city is higher : {summary[3]:,}")
print(f"Hours zone is higher : {summary[4]:,}")

print("\nDST HOURS")
print(con.execute("""
    SELECT *
    FROM zone
    WHERE pickup_hour IN (
        TIMESTAMP '2024-03-10 02:00:00',
        TIMESTAMP '2025-03-09 02:00:00'
    )
""").fetchdf().to_string(index=False))

con.close()
