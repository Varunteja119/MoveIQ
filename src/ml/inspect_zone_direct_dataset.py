from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ml/processed"

con = duckdb.connect()

for split in ["train", "validation", "test"]:
    path = DATA / f"zone_direct_{split}.parquet"

    print("\n" + "=" * 65)
    print(f"{split.upper()} DATASET")
    print("=" * 65)

    print("\nSchema:")
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
    ).fetchall()

    for column in schema:
        print(f"{column[0]:25} {column[1]}")

    print("\nRow count and null checks:")
    result = con.execute(f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(*) FILTER (WHERE zone_id IS NULL) AS null_zone,
            COUNT(*) FILTER (WHERE borough IS NULL) AS null_borough,
            COUNT(*) FILTER (WHERE service_zone IS NULL) AS null_service_zone,
            COUNT(*) FILTER (WHERE actual_demand IS NULL) AS null_target,
            COUNT(DISTINCT zone_id) AS zones
        FROM read_parquet('{path.as_posix()}')
    """).fetchone()

    print({
        "rows": result[0],
        "null_zone": result[1],
        "null_borough": result[2],
        "null_service_zone": result[3],
        "null_target": result[4],
        "zones": result[5],
    })

    print("\nSample:")
    sample = con.execute(f"""
        SELECT *
        FROM read_parquet('{path.as_posix()}')
        LIMIT 3
    """).fetchdf()

    print(sample.to_string(index=False))

con.close()
