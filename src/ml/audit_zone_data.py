from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "ml" / "raw"
LOOKUP_FILE = ROOT / "data" / "reference" / "taxi_zone_lookup.csv"

con = duckdb.connect()

files = sorted(DATA_DIR.glob("yellow_tripdata_*.parquet"))

print("=" * 70)
print("MOVEIQ - ZONE-LEVEL DATA READINESS AUDIT")
print("=" * 70)

print(f"\nMonthly Parquet files found: {len(files)}")

if not files:
    raise FileNotFoundError("No historical monthly Parquet files found.")

sample_file = files[0].as_posix()

print("\nSample Parquet schema:")
schema = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{sample_file}')"
).fetchdf()

print(schema.to_string(index=False))

lookup_path = LOOKUP_FILE.as_posix()

print("\nTaxi zone lookup:")
lookup = con.execute(
    f"SELECT * FROM read_csv_auto('{lookup_path}') LIMIT 10"
).fetchdf()

print(lookup.to_string(index=False))

print("\nHistorical zone coverage:")

all_files = [f.as_posix() for f in files]
file_list = "[" + ",".join("'" + f.replace("'", "''") + "'" for f in all_files) + "]"

coverage = con.execute(f"""
    SELECT
        COUNT(*) AS total_records,
        COUNT(DISTINCT PULocationID) AS unique_pickup_zones,
        COUNT(DISTINCT DOLocationID) AS unique_dropoff_zones,
        MIN(tpep_pickup_datetime) AS first_pickup,
        MAX(tpep_pickup_datetime) AS last_pickup
    FROM read_parquet({file_list}, union_by_name=true)
    WHERE PULocationID IS NOT NULL
""").fetchdf()

print(coverage.to_string(index=False))

print("\nTop 20 pickup zones across the full historical dataset:")

top_zones = con.execute(f"""
    SELECT
        t.PULocationID AS zone_id,
        z.Zone AS zone_name,
        z.Borough AS borough,
        COUNT(*) AS total_pickups
    FROM read_parquet({file_list}, union_by_name=true) t
    LEFT JOIN read_csv_auto('{lookup_path}') z
        ON t.PULocationID = z.LocationID
    WHERE t.PULocationID IS NOT NULL
    GROUP BY 1, 2, 3
    ORDER BY total_pickups DESC
    LIMIT 20
""").fetchdf()

print(top_zones.to_string(index=False))

print("\nZone-level data readiness audit complete.")

con.close()
