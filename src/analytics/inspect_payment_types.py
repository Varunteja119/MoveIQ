import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILE = (
    ROOT
    / "data"
    / "processed"
    / "yellow_tripdata_2025-01_cleaned.parquet"
)

con = duckdb.connect()

result = con.execute("""
    SELECT
        payment_type,
        COUNT(*) AS trip_count
    FROM read_parquet(?)
    GROUP BY payment_type
    ORDER BY payment_type
""", [str(FILE)]).df()

print("\nPAYMENT TYPE CODE INSPECTION")
print("=" * 45)
print(result.to_string(index=False))

con.close()
