from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/yellow_tripdata_2025-01_cleaned.parquet"
LOOKUP_FILE = ROOT / "data/reference/taxi_zone_lookup.csv"
OUTPUT_DIR = ROOT / "data/processed/eda"
CHART_DIR = ROOT / "reports/figures"

BATCH_SIZE = 100_000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(INPUT_FILE)

    # Aggregate pickup counts by zone and hour
    results = []

    print("\nAnalyzing pickup demand by zone and hour...\n")

    for batch_number, batch in enumerate(
        parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["tpep_pickup_datetime", "PULocationID"],
        ),
        start=1,
    ):
        df = batch.to_pandas()

        df["hour"] = df["tpep_pickup_datetime"].dt.hour

        grouped = (
            df.groupby(["PULocationID", "hour"])
            .size()
            .reset_index(name="trip_count")
        )

        results.append(grouped)

        print(f"Processed batch {batch_number}")

    combined = pd.concat(results, ignore_index=True)

    demand = (
        combined.groupby(["PULocationID", "hour"])["trip_count"]
        .sum()
        .reset_index()
    )

    # Find the 15 busiest pickup zones overall
    top_zones = (
        demand.groupby("PULocationID")["trip_count"]
        .sum()
        .nlargest(15)
        .index
    )

    demand = demand[demand["PULocationID"].isin(top_zones)]

    # Add neighborhood names
    lookup = pd.read_csv(LOOKUP_FILE)

    demand = demand.merge(
        lookup[["LocationID", "Zone", "Borough"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )

    # Build the zone-hour matrix
    matrix = demand.pivot_table(
        index="Zone",
        columns="hour",
        values="trip_count",
        aggfunc="sum",
        fill_value=0,
    )

    matrix = matrix.reindex(columns=range(24), fill_value=0)

    # Average monthly counts per calendar day
    matrix = matrix / 31

    matrix.to_csv(
        OUTPUT_DIR / "pickup_zone_hour_average.csv"
    )

    # Sort zones by total demand
    matrix = matrix.loc[
        matrix.sum(axis=1).sort_values(ascending=False).index
    ]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 9))

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_title(
        "Average Daily Pickup Demand by Neighborhood and Hour",
        fontsize=16,
        pad=15,
    )

    ax.set_xlabel("Pickup Hour")
    ax.set_ylabel("Pickup Neighborhood")

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Average Trips per Day")

    ax.set_xticks(
        [x - 0.5 for x in range(1, 24)],
        minor=True,
    )
    ax.grid(which="minor", color="white", linewidth=0.5)

    plt.tight_layout()

    output_file = CHART_DIR / "pickup_zone_hour_heatmap.png"

    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nMOVEIQ ZONE-HOUR ANALYSIS COMPLETED")
    print(f"Top zones analyzed: {len(matrix)}")
    print(f"Heatmap: {output_file}")
    print(f"CSV: {OUTPUT_DIR / 'pickup_zone_hour_average.csv'}")


if __name__ == "__main__":
    main()
