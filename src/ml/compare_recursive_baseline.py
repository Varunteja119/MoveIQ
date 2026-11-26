from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = ROOT / "data" / "ml" / "processed" / "hourly_demand_2024_2025.parquet"
PREDICTIONS_PATH = ROOT / "reports" / "ml" / "recursive_24h_backtest_predictions.csv"
OUTPUT_DIR = ROOT / "reports" / "ml"


def calculate_metrics(actual, predicted):
    errors = actual - predicted

    return {
        "MAE": np.abs(errors).mean(),
        "RMSE": np.sqrt(np.square(errors).mean()),
    }


def main():
    history = pd.read_parquet(HISTORY_PATH)
    history["pickup_hour"] = pd.to_datetime(history["pickup_hour"])

    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions["forecast_timestamp"] = pd.to_datetime(
        predictions["forecast_timestamp"]
    )

    actual_lookup = dict(
        zip(history["pickup_hour"], history["trip_count"])
    )

    # Previous-week demand is known at the forecast origin.
    predictions["previous_week_timestamp"] = (
        predictions["forecast_timestamp"] - pd.Timedelta(hours=168)
    )

    predictions["previous_week_prediction"] = (
        predictions["previous_week_timestamp"].map(actual_lookup)
    )

    if predictions["previous_week_prediction"].isna().any():
        raise ValueError("Some previous-week baseline values are unavailable.")

    overall_rows = []

    for model_name, prediction_column in [
        ("Recursive XGBoost", "predicted_pickups"),
        ("Previous-week baseline", "previous_week_prediction"),
    ]:
        metrics = calculate_metrics(
            predictions["actual_pickups"].to_numpy(),
            predictions[prediction_column].to_numpy(),
        )

        overall_rows.append({
            "model": model_name,
            **metrics,
        })

    overall = pd.DataFrame(overall_rows)

    horizon_rows = []

    for horizon, group in predictions.groupby("hours_ahead"):
        xgb_metrics = calculate_metrics(
            group["actual_pickups"].to_numpy(),
            group["predicted_pickups"].to_numpy(),
        )

        baseline_metrics = calculate_metrics(
            group["actual_pickups"].to_numpy(),
            group["previous_week_prediction"].to_numpy(),
        )

        horizon_rows.append({
            "hours_ahead": horizon,
            "xgboost_MAE": xgb_metrics["MAE"],
            "previous_week_MAE": baseline_metrics["MAE"],
            "xgboost_RMSE": xgb_metrics["RMSE"],
            "previous_week_RMSE": baseline_metrics["RMSE"],
        })

    by_horizon = pd.DataFrame(horizon_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall.to_csv(
        OUTPUT_DIR / "recursive_baseline_overall.csv",
        index=False,
    )

    by_horizon.to_csv(
        OUTPUT_DIR / "recursive_baseline_by_horizon.csv",
        index=False,
    )

    print("\nMOVEIQ: RECURSIVE MODEL VS PREVIOUS-WEEK BASELINE")
    print("=" * 75)
    print(overall.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\nCOMPARISON BY FORECAST HORIZON")
    print("-" * 75)
    print(by_horizon.to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}",
    ))

    xgb_mae = overall.loc[
        overall["model"] == "Recursive XGBoost", "MAE"
    ].iloc[0]

    baseline_mae = overall.loc[
        overall["model"] == "Previous-week baseline", "MAE"
    ].iloc[0]

    print("\nCONCLUSION")
    print("-" * 75)

    if xgb_mae < baseline_mae:
        improvement = (baseline_mae - xgb_mae) / baseline_mae * 100
        print(
            f"Recursive XGBoost beats the previous-week baseline "
            f"by {improvement:.2f}% in overall MAE."
        )
    else:
        degradation = (xgb_mae - baseline_mae) / baseline_mae * 100
        print(
            f"Recursive XGBoost is {degradation:.2f}% worse "
            f"than the previous-week baseline in overall MAE."
        )

    print("\nSaved comparison reports in reports/ml/")


if __name__ == "__main__":
    main()
