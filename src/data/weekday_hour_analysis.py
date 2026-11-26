from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "yellow_tripdata_2025-01_cleaned.parquet"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "eda"
CHART_DIR = ROOT / "reports" / "figures"

BATCH_SIZE = 100_000

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(INPUT_FILE)

    # Count trips for each weekday and hour
    demand = np.zeros((7, 24), dtype=np.int64)

    # Count how many calendar dates each weekday occurs
    dates = pd.date_range("2025-01-01", "2025-01-31")
    weekday_occurrences = (
        pd.Series(dates.dayofweek)
        .value_counts()
        .sort_index()
        .reindex(range(7), fill_value=0)
        .to_numpy()
    )

    total_rows = 0

    print("\nBuilding weekday-hour demand matrix...\n")

    for batch_number, batch in enumerate(
        parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["tpep_pickup_datetime"],
        ),
        start=1,
    ):

        df = batch.to_pandas()
        pickup = df["tpep_pickup_datetime"]

        weekday = pickup.dt.day_of_week.to_numpy()
        hour = pickup.dt.hour.to_numpy()

        np.add.at(demand, (weekday, hour), 1)

        total_rows += len(df)

        print(
            f"Processed batch {batch_number} | "
            f"Rows: {total_rows:,}"
        )

    # Average trips per occurrence of each weekday
    average_demand = (
        demand / weekday_occurrences[:, np.newaxis]
    )

    # Save raw demand counts
    raw_df = pd.DataFrame(
        demand,
        index=WEEKDAYS,
        columns=range(24),
    )
    raw_df.index.name = "weekday"
    raw_df.to_csv(OUTPUT_DIR / "weekday_hour_demand_counts.csv")

    # Save average demand
    average_df = pd.DataFrame(
        average_demand,
        index=WEEKDAYS,
        columns=range(24),
    )
    average_df.index.name = "weekday"
    average_df.to_csv(OUTPUT_DIR / "weekday_hour_average_demand.csv")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 7))

    image = ax.imshow(
        average_demand,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_title(
        "NYC Yellow Taxi Average Demand by Weekday and Hour",
        fontsize=16,
        pad=15,
    )

    ax.set_xlabel("Pickup Hour", fontsize=12)
    ax.set_ylabel("Day of Week", fontsize=12)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{hour:02d}:00" for hour in range(24)])

    ax.set_yticks(range(7))
    ax.set_yticklabels(WEEKDAYS)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Average Trips per Weekday Occurrence")

    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    output_image = CHART_DIR / "weekday_hour_heatmap.png"

    plt.savefig(
        output_image,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # Print peak demand for each weekday
    print("\nPEAK DEMAND BY WEEKDAY")
    print("=" * 55)

    for i, day in enumerate(WEEKDAYS):
        peak_hour = int(np.argmax(average_demand[i]))
        peak_value = average_demand[i, peak_hour]

        print(
            f"{day:<12} "
            f"Peak: {peak_hour:02d}:00 | "
            f"Average trips: {peak_value:,.0f}"
        )

    print("\n" + "=" * 55)
    print("MOVEIQ HEATMAP COMPLETED")
    print("=" * 55)

    print(f"Rows analyzed: {total_rows:,}")
    print(f"Heatmap: {output_image}")
    print(f"CSV files: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
