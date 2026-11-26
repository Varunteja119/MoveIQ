from pathlib import Path
import duckdb

INPUT_FILE = Path("data/ml/processed/hourly_demand_features.parquet")
OUTPUT_DIR = Path("data/ml/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
source = INPUT_FILE.resolve().as_posix()

# Use rows with all required model features.
df = con.execute(f"""
    SELECT *
    FROM read_parquet('{source}')
    WHERE
        lag_1 IS NOT NULL
        AND lag_24 IS NOT NULL
        AND lag_168 IS NOT NULL
        AND rolling_mean_24 IS NOT NULL
        AND rolling_mean_168 IS NOT NULL
    ORDER BY pickup_hour
""").fetchdf()

con.close()

total = len(df)
train_end = int(total * 0.70)
validation_end = int(total * 0.85)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:validation_end].copy()
test = df.iloc[validation_end:].copy()

train.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
validation.to_parquet(OUTPUT_DIR / "validation.parquet", index=False)
test.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

print("=" * 65)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 65)

for name, part in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    print(f"\n{name}")
    print(f"Records: {len(part):,}")
    print(f"Start:   {part['pickup_hour'].min()}")
    print(f"End:     {part['pickup_hour'].max()}")

print("\nTotal usable records:", total)
print("Original feature dataset remains unchanged.")
print("\nSaved all three datasets successfully.")
