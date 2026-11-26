from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]

EDA_DIR = ROOT / "data" / "processed" / "eda"
CHART_DIR = ROOT / "reports" / "figures"

CHART_DIR.mkdir(parents=True, exist_ok=True)

def thousands(x, position):
    return f"{x / 1000:.0f}K"


def main():

    # Hourly average per calendar day
    hourly = pd.read_csv(EDA_DIR / "hourly_demand.csv")
    hourly["average_trips_per_day"] = hourly["trip_count"] / 31

    plt.figure(figsize=(12, 5))
    plt.bar(
        hourly["pickup_hour"],
        hourly["average_trips_per_day"],
    )
    plt.title("Average NYC Yellow Taxi Demand by Hour")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Average Trips per Day")
    plt.xticks(range(24))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "hourly_demand_average.png", dpi=200)
    plt.close()

    # Weekday average per occurrence
    weekday = pd.read_csv(EDA_DIR / "weekday_demand.csv")

    weekday_occurrences = {
        "Monday": 4,
        "Tuesday": 4,
        "Wednesday": 5,
        "Thursday": 5,
        "Friday": 5,
        "Saturday": 4,
        "Sunday": 4,
    }

    weekday["days_in_month"] = weekday["weekday"].map(
        weekday_occurrences
    )

    weekday["average_trips_per_day"] = (
        weekday["trip_count"] / weekday["days_in_month"]
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday["weekday"] = pd.Categorical(
        weekday["weekday"],
        categories=weekday_order,
        ordered=True,
    )

    weekday = weekday.sort_values("weekday")

    plt.figure(figsize=(10, 5))
    plt.bar(
        weekday["weekday"],
        weekday["average_trips_per_day"],
    )
    plt.title("Average NYC Yellow Taxi Demand by Weekday")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Trips per Occurrence")
    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "weekday_demand_average.png", dpi=200)
    plt.close()

    # Neighborhood charts with readable K-axis
    for filename, title, output in [
        (
            "pickup_zones_named.csv",
            "Top 10 Pickup Neighborhoods",
            "pickup_neighborhoods_readable.png",
        ),
        (
            "dropoff_zones_named.csv",
            "Top 10 Drop-off Neighborhoods",
            "dropoff_neighborhoods_readable.png",
        ),
    ]:

        df = pd.read_csv(EDA_DIR / filename).head(10)
        df = df.sort_values("trip_count")

        plt.figure(figsize=(11, 7))
        plt.barh(df["zone_name"], df["trip_count"])
        plt.title(f"NYC Yellow Taxi {title} - January 2025")
        plt.xlabel("Total Trips (Thousands)")
        plt.ylabel("Neighborhood")
        plt.gca().xaxis.set_major_formatter(
            FuncFormatter(thousands)
        )
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHART_DIR / output, dpi=200)
        plt.close()

    print("\nImproved charts generated successfully.")

    print("\nAverage weekday demand:")
    print(
        weekday[
            ["weekday", "average_trips_per_day"]
        ].to_string(index=False)
    )

    print(f"\nCharts saved in: {CHART_DIR}")


if __name__ == "__main__":
    main()
