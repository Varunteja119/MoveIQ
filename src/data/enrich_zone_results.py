from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EDA_DIR = ROOT / "data" / "processed" / "eda"
LOOKUP_FILE = ROOT / "data" / "reference" / "taxi_zone_lookup.csv"


def enrich_zones(input_file, id_column, output_file):
    counts = pd.read_csv(input_file)
    zones = pd.read_csv(LOOKUP_FILE)

    zones = zones.rename(columns={
        "LocationID": "location_id",
        "Zone": "zone_name",
        "Borough": "borough",
        "service_zone": "service_zone",
    })

    result = counts.merge(
        zones[["location_id", "zone_name", "borough", "service_zone"]],
        left_on=id_column,
        right_on="location_id",
        how="left",
    )

    result = result.rename(columns={
        id_column: "location_id",
    })

    result = result.drop(columns=["location_id_y"], errors="ignore")

    result = result.rename(columns={
        "location_id_x": "location_id",
    })

    result.insert(
        0,
        "rank",
        range(1, len(result) + 1),
    )

    result.to_csv(output_file, index=False)

    print(f"\n{output_file.name}")
    print(result.head(10).to_string(index=False))


def main():
    if not LOOKUP_FILE.exists():
        raise FileNotFoundError(LOOKUP_FILE)

    enrich_zones(
        EDA_DIR / "pickup_zones.csv",
        "pickup_location_id",
        EDA_DIR / "pickup_zones_named.csv",
    )

    enrich_zones(
        EDA_DIR / "dropoff_zones.csv",
        "dropoff_location_id",
        EDA_DIR / "dropoff_zones_named.csv",
    )

    print("\nZone enrichment completed.")


if __name__ == "__main__":
    main()
