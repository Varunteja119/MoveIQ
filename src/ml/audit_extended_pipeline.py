from pathlib import Path
import duckdb

ROOT = Path.cwd()
DATA = ROOT / "data" / "ml" / "processed"

FILES = {
    "city_features": DATA / "hourly_demand_features_2024_2026.parquet",
    "zone_features": DATA / "zone_hourly_demand_features_2024_2026.parquet",
    "city_train": DATA / "train_2024_2026.parquet",
    "city_validation": DATA / "validation_2024_2026.parquet",
    "city_test": DATA / "test_2024_2026.parquet",
    "zone_train": DATA / "zone_train_2024_2026.parquet",
    "zone_validation": DATA / "zone_validation_2024_2026.parquet",
    "zone_test": DATA / "zone_test_2024_2026.parquet",
}

missing = [name for name, path in FILES.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing expected files: {missing}")

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET threads=4")

for name, path in FILES.items():
    p = str(path).replace("\\", "/").replace("'", "''")
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p}')")

print("\n" + "=" * 78)
print("MOVEIQ EXTENDED PIPELINE — READ-ONLY DATA AUDIT")
print("=" * 78)

print("\n1. FEATURE DATASET SCHEMAS")
for name in ["city_features", "zone_features"]:
    print(f"\n{name}:")
    for row in con.execute(f"DESCRIBE SELECT * FROM {name}").fetchall():
        print(f"  {row[0]}: {row[1]}")

print("\n2. TOTALS BEFORE FEATURE NULL FILTERING")
for name in ["city_features", "zone_features"]:
    result = con.execute(
        f"""SELECT COUNT(*), SUM(trip_count),
                   MIN(pickup_hour), MAX(pickup_hour)
            FROM {name}"""
    ).fetchone()
    print(f"{name}: rows={result[0]:,}, pickups={result[1]:,}, "
          f"range={result[2]} to {result[3]}")

print("\n3. NULL COUNTS FOR MODEL HISTORY FEATURES")
features = ["lag_1", "lag_24", "lag_168",
            "rolling_mean_24", "rolling_mean_168"]

for name in ["city_features", "zone_features"]:
    print(f"\n{name}:")
    expressions = ", ".join(
        f"COUNT(*) FILTER (WHERE {col} IS NULL) AS {col}_nulls"
        for col in features
    )
    result = con.execute(f"SELECT {expressions} FROM {name}").fetchone()
    for col, value in zip(features, result):
        print(f"  {col}: {value:,} nulls")

print("\n4. CITYWIDE VS ZONE TOTALS AT EACH TIMESTAMP")
result = con.execute("""
    WITH city AS (
        SELECT pickup_hour, SUM(trip_count) AS city_total
        FROM city_features GROUP BY pickup_hour
    ),
    zone AS (
        SELECT pickup_hour, SUM(trip_count) AS zone_total
        FROM zone_features GROUP BY pickup_hour
    ),
    comparison AS (
        SELECT c.pickup_hour, c.city_total, z.zone_total,
               z.zone_total - c.city_total AS difference
        FROM city c JOIN zone z USING (pickup_hour)
    )
    SELECT COUNT(*),
           SUM(city_total),
           SUM(zone_total),
           SUM(difference),
           COUNT(*) FILTER (WHERE difference > 0),
           COUNT(*) FILTER (WHERE difference < 0),
           MAX(ABS(difference))
    FROM comparison
""").fetchone()

labels = [
    "Common timestamps", "Citywide pickups", "Zone pickups",
    "Total zone-minus-city difference", "Hours zone > city",
    "Hours zone < city", "Maximum absolute hourly difference"
]
for label, value in zip(labels, result):
    print(f"  {label}: {value:,}")

print("\nLargest hourly discrepancies:")
rows = con.execute("""
    WITH city AS (
        SELECT pickup_hour, SUM(trip_count) AS city_total
        FROM city_features GROUP BY pickup_hour
    ),
    zone AS (
        SELECT pickup_hour, SUM(trip_count) AS zone_total
        FROM zone_features GROUP BY pickup_hour
    )
    SELECT c.pickup_hour, c.city_total, z.zone_total,
           z.zone_total - c.city_total AS difference
    FROM city c JOIN zone z USING (pickup_hour)
    WHERE z.zone_total <> c.city_total
    ORDER BY ABS(z.zone_total - c.city_total) DESC
    LIMIT 15
""").fetchall()

if not rows:
    print("  No hourly discrepancies.")
else:
    for row in rows:
        print(f"  {row[0]} | city={row[1]:,} | zone={row[2]:,} "
              f"| difference={row[3]:+,}")

print("\n5. TOTALS AFTER REQUIRING ALL FIVE HISTORY FEATURES")
valid_condition = " AND ".join(f"{col} IS NOT NULL" for col in features)

for name in ["city_features", "zone_features"]:
    result = con.execute(f"""
        SELECT COUNT(*), SUM(trip_count),
               MIN(pickup_hour), MAX(pickup_hour)
        FROM {name}
        WHERE {valid_condition}
    """).fetchone()
    print(f"{name}: rows={result[0]:,}, pickups={result[1]:,}, "
          f"range={result[2]} to {result[3]}")

print("\n6. CHRONOLOGICAL SPLIT TOTALS")
for name in [
    "city_train", "city_validation", "city_test",
    "zone_train", "zone_validation", "zone_test"
]:
    result = con.execute(
        f"""SELECT COUNT(*), SUM(trip_count),
                   MIN(pickup_hour), MAX(pickup_hour)
            FROM {name}"""
    ).fetchone()
    print(f"{name}: rows={result[0]:,}, pickups={result[1]:,}, "
          f"range={result[2]} to {result[3]}")

print("\n7. DIRECT DATASET SANITY CHECK")
direct_files = [
    "direct_train_2024_2026.parquet",
    "direct_validation_2024_2026.parquet",
    "direct_test_2024_2026.parquet",
    "direct_zone_train_2024_2026.parquet",
    "direct_zone_validation_2024_2026.parquet",
    "direct_zone_test_2024_2026.parquet",
]

for filename in direct_files:
    path = DATA / filename
    if not path.exists():
        print(f"{filename}: MISSING")
        continue
    p = str(path).replace("\\", "/").replace("'", "''")
    result = con.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT origin_timestamp),
               COUNT(DISTINCT hours_ahead),
               MIN(origin_timestamp), MAX(origin_timestamp),
               MIN(target_timestamp), MAX(target_timestamp)
        FROM read_parquet('{p}')
    """).fetchone()
    print(f"{filename}: rows={result[0]:,}, "
          f"origins={result[1]:,}, horizons={result[2]}, "
          f"origin_range={result[3]} to {result[4]}, "
          f"target_range={result[5]} to {result[6]}")

print("\n" + "=" * 78)
print("AUDIT COMPLETE — NO FILES MODIFIED")
print("=" * 78)

con.close()
