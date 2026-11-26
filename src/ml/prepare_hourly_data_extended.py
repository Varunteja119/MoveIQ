from pathlib import Path
from datetime import date
import re
import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/ml/raw"
OUTPUT_DIR = ROOT / "data/ml/processed"
OUTPUT_FILE = OUTPUT_DIR / "hourly_demand_2024_2026.parquet"

START_DATE = date(2024, 1, 1)


def discover_months():
    pattern = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet$")
    discovered = []

    for path in RAW_DIR.glob("yellow_tripdata_*.parquet"):
        match = pattern.fullmatch(path.name)
        if match:
            year, month = map(int, match.groups())
            if not 1 <= month <= 12:
                raise ValueError(f"Invalid month in filename: {path.name}")
            discovered.append((year, month, path))

    discovered.sort(key=lambda item: (item[0], item[1]))

    if not discovered:
        raise RuntimeError("No monthly TLC Parquet files found.")

    expected = []
    year, month = 2024, 1
    latest_year, latest_month, _ = discovered[-1]

    while (year, month) <= (latest_year, latest_month):
        expected.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1

    actual = [(year, month) for year, month, _ in discovered]

    missing = sorted(set(expected) - set(actual))
    if missing:
        raise RuntimeError(f"Missing monthly files: {missing}")

    for _, _, path in discovered:
        with path.open("rb") as file:
            if file.read(4) != b"PAR1":
                raise RuntimeError(f"Invalid Parquet signature: {path.name}")

    end_year, end_month = latest_year, latest_month + 1
    if end_month == 13:
        end_year += 1
        end_month = 1

    return discovered, date(end_year, end_month, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files, end_date = discover_months()

    file_list = "[" + ", ".join(
        "'" + str(path.resolve()).replace("\\", "/") + "'"
        for _, _, path in files
    ) + "]"

    end_timestamp = end_date.isoformat()

    print("=" * 75)
    print("MOVEIQ — EXTENDED CITYWIDE HOURLY DATA BUILDER")
    print("=" * 75)
    print(f"Monthly files: {len(files)}")
    print(f"Date range: {START_DATE} to {end_date} (exclusive)")
    print(f"Output: {OUTPUT_FILE}")

    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET preserve_insertion_order = false")

    print("\nAggregating all monthly files...")

    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('hour', tpep_pickup_datetime) AS pickup_hour,
                COUNT(*)::BIGINT AS trip_count
            FROM read_parquet({file_list}, union_by_name = true)
            WHERE
                tpep_pickup_datetime >= TIMESTAMP '{START_DATE.isoformat()}'
                AND tpep_pickup_datetime < TIMESTAMP '{end_timestamp}'
            GROUP BY 1
            ORDER BY 1
        )
        TO '{OUTPUT_FILE.resolve().as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    output_path = OUTPUT_FILE.resolve().as_posix()

    summary = con.execute(f"""
        SELECT
            COUNT(*) AS hourly_records,
            MIN(pickup_hour) AS first_hour,
            MAX(pickup_hour) AS last_hour,
            SUM(trip_count) AS total_trips,
            ROUND(AVG(trip_count), 2) AS average_hourly_trips,
            MIN(trip_count) AS minimum_hourly_trips,
            MAX(trip_count) AS maximum_hourly_trips
        FROM read_parquet('{output_path}')
    """).fetchdf()

    duplicates = con.execute(f"""
        SELECT COUNT(*) - COUNT(DISTINCT pickup_hour)
        FROM read_parquet('{output_path}')
    """).fetchone()[0]

    print("\n" + "=" * 75)
    print("EXTENDED CITYWIDE DATASET SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))
    print(f"\nDuplicate timestamps: {duplicates}")

    if duplicates != 0:
        raise RuntimeError("Duplicate hourly timestamps detected.")

    if summary.iloc[0]["hourly_records"] == 0:
        raise RuntimeError("The generated dataset is empty.")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")
    print("\nSUCCESS: Extended citywide hourly dataset created.")

    con.close()


if __name__ == "__main__":
    main()