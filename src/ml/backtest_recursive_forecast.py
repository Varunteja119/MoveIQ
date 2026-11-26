from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.forecast_service import MoveIQForecastService


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "ml" / "processed" / "test.parquet"
OUTPUT_DIR = ROOT / "reports" / "ml"


def main():
    service = MoveIQForecastService()
    full_history = service.history.copy()

    test_data = pd.read_parquet(TEST_PATH)
    test_timestamps = set(
        pd.to_datetime(test_data["pickup_hour"]).tolist()
    )

    # Start immediately before the first held-out test observation.
    first_test_timestamp = min(test_timestamps)
    matching_indices = full_history.index[
        full_history["pickup_hour"] == first_test_timestamp
    ].tolist()

    if not matching_indices:
        raise ValueError("Test period does not match historical data.")

    first_test_index = matching_indices[0]
    origin_indices = range(
        first_test_index - 1,
        len(full_history) - 24,
        24,
    )

    all_results = []

    for origin_index in origin_indices:
        historical_cutoff = full_history.iloc[origin_index]["pickup_hour"]

        actual_window = full_history.iloc[
            origin_index + 1:origin_index + 25
        ].copy()

        if len(actual_window) != 24:
            continue

        if not all(ts in test_timestamps for ts in actual_window["pickup_hour"]):
            continue

        # Give the model only information available at this forecast origin.
        service.history = full_history.iloc[:origin_index + 1].copy()

        predictions = service.forecast_next_24_hours()

        predicted_times = pd.to_datetime(
            predictions["forecast_timestamp"]
        ).reset_index(drop=True)

        actual_times = pd.to_datetime(
            actual_window["pickup_hour"]
        ).reset_index(drop=True)

        if not predicted_times.equals(actual_times):
            raise ValueError(
                f"Forecast timestamps do not align after {historical_cutoff}."
            )

        for i in range(24):
            actual = float(actual_window.iloc[i]["trip_count"])
            predicted = float(predictions.iloc[i]["predicted_pickups"])

            all_results.append({
                "forecast_origin": historical_cutoff,
                "forecast_timestamp": predicted_times.iloc[i],
                "hours_ahead": i + 1,
                "actual_pickups": actual,
                "predicted_pickups": predicted,
                "absolute_error": abs(actual - predicted),
                "squared_error": (actual - predicted) ** 2,
            })

    if not all_results:
        raise RuntimeError("No complete 24-hour backtest windows were found.")

    results = pd.DataFrame(all_results)

    mae = results["absolute_error"].mean()
    rmse = np.sqrt(results["squared_error"].mean())

    horizon_metrics = (
        results.groupby("hours_ahead")
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
            observations=("absolute_error", "count"),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "recursive_24h_backtest_predictions.csv",
        index=False,
    )

    horizon_metrics.to_csv(
        OUTPUT_DIR / "recursive_24h_backtest_metrics.csv",
        index=False,
    )

    print("\nMOVEIQ RECURSIVE 24-HOUR BACKTEST")
    print("=" * 65)
    print(f"Forecast windows : {results['forecast_origin'].nunique()}")
    print(f"Total predictions: {len(results)}")
    print(f"Overall MAE      : {mae:,.2f} pickups")
    print(f"Overall RMSE     : {rmse:,.2f} pickups")

    print("\nERROR BY FORECAST HORIZON")
    print("-" * 65)

    selected = horizon_metrics[
        horizon_metrics["hours_ahead"].isin([1, 2, 3, 6, 12, 18, 21, 24])
    ]

    print(selected.to_string(index=False))

    print("\nERROR TREND")
    print("-" * 65)

    first_six = horizon_metrics.head(6)["MAE"].mean()
    last_six = horizon_metrics.tail(6)["MAE"].mean()

    print(f"Average MAE, hours 1-6  : {first_six:,.2f}")
    print(f"Average MAE, hours 19-24: {last_six:,.2f}")

    if last_six > first_six:
        print("Later forecasts have higher average error.")
    else:
        print("Later forecasts do not have higher average error in this test.")

    print("\nSaved:")
    print(OUTPUT_DIR / "recursive_24h_backtest_predictions.csv")
    print(OUTPUT_DIR / "recursive_24h_backtest_metrics.csv")


if __name__ == "__main__":
    main()
