from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.forecast_service import MoveIQForecastService


ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data/ml/processed/hourly_demand_2024_2025.parquet"
TEST_PATH = ROOT / "data/ml/processed/test.parquet"

DIRECT_MODEL_PATH = ROOT / "models/moveiq_direct_demand_forecaster.joblib"
RECURSIVE_MODEL_PATH = ROOT / "models/moveiq_demand_forecaster.joblib"

OUTPUT_DIR = ROOT / "reports/ml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECT_FEATURES = [
    "target_hour",
    "target_day_of_week",
    "target_is_weekend",
    "target_month",
    "hours_ahead",
    "origin_demand",
    "origin_lag_24",
    "origin_lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "target_lag_24",
    "target_lag_168",
]


def calculate_metrics(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
    }


print("Loading historical data and models...")

history = pd.read_parquet(HISTORY_PATH)
test = pd.read_parquet(TEST_PATH)

history["pickup_hour"] = pd.to_datetime(history["pickup_hour"])
test["pickup_hour"] = pd.to_datetime(test["pickup_hour"])

history = history.sort_values("pickup_hour").reset_index(drop=True)
test = test.sort_values("pickup_hour").reset_index(drop=True)

direct_model = joblib.load(DIRECT_MODEL_PATH)

recursive_service = MoveIQForecastService(
    history_path=HISTORY_PATH,
    model_path=RECURSIVE_MODEL_PATH,
)

# Confirm the model expects the features we are constructing.
model_features = list(direct_model.feature_names_in_)

if set(model_features) != set(DIRECT_FEATURES):
    raise ValueError(
        "Direct model feature mismatch.\n"
        f"Expected from model: {model_features}\n"
        f"Constructed here: {DIRECT_FEATURES}"
    )

history_lookup = dict(
    zip(history["pickup_hour"], history["trip_count"])
)

history_index = history.set_index("pickup_hour")["trip_count"]

test_timestamps = test["pickup_hour"].reset_index(drop=True)

# Select daily-spaced origins, leaving enough test observations
# for a complete 24-hour forecast after each origin.
origins = test_timestamps.iloc[::24].tolist()

valid_origins = []

for origin in origins:
    targets = [
        origin + pd.Timedelta(hours=h)
        for h in range(1, 25)
    ]

    if all(timestamp in history_lookup for timestamp in targets):
        valid_origins.append(origin)

print(f"\nDaily test forecast origins: {len(valid_origins)}")

if not valid_origins:
    raise ValueError("No valid 24-hour test forecast windows found.")

all_results = []

print("\nGenerating predictions...")

for index, origin in enumerate(valid_origins, start=1):

    # Only information available at the forecast starting time.
    available_history = history[
        history["pickup_hour"] <= origin
    ].copy()

    if available_history.empty:
        raise ValueError(f"No history available at {origin}")

    origin_demand = float(
        history_lookup[origin]
    )

    origin_lag_24_time = origin - pd.Timedelta(hours=24)
    origin_lag_168_time = origin - pd.Timedelta(hours=168)

    if (
        origin_lag_24_time not in history_lookup
        or origin_lag_168_time not in history_lookup
    ):
        raise ValueError(f"Missing origin lag at {origin}")

    prior_24 = available_history[
        (available_history["pickup_hour"] < origin)
        & (
            available_history["pickup_hour"]
            >= origin - pd.Timedelta(hours=24)
        )
    ]["trip_count"]

    prior_168 = available_history[
        (available_history["pickup_hour"] < origin)
        & (
            available_history["pickup_hour"]
            >= origin - pd.Timedelta(hours=168)
        )
    ]["trip_count"]

    if prior_24.empty or prior_168.empty:
        raise ValueError(f"Missing rolling history at {origin}")

    # Recursive model receives history only up to this origin.
    recursive_service.history = available_history[
        ["pickup_hour", "trip_count"]
    ].copy().reset_index(drop=True)

    recursive_predictions = (
        recursive_service.forecast_next_24_hours()
    )

    for hours_ahead in range(1, 25):

        target = origin + pd.Timedelta(hours=hours_ahead)

        target_lag_24_time = target - pd.Timedelta(hours=24)
        target_lag_168_time = target - pd.Timedelta(hours=168)

        if (
            target_lag_24_time not in history_lookup
            or target_lag_168_time not in history_lookup
        ):
            raise ValueError(f"Missing target lag at {target}")

        target_day = (target.day_of_week + 1) % 7

        features = {
            "target_hour": target.hour,
            "target_day_of_week": target_day,
            "target_is_weekend": int(target_day in [0, 6]),
            "target_month": target.month,
            "hours_ahead": hours_ahead,
            "origin_demand": origin_demand,
            "origin_lag_24": float(history_lookup[origin_lag_24_time]),
            "origin_lag_168": float(history_lookup[origin_lag_168_time]),
            "rolling_mean_24": float(prior_24.mean()),
            "rolling_mean_168": float(prior_168.mean()),
            "target_lag_24": float(history_lookup[target_lag_24_time]),
            "target_lag_168": float(history_lookup[target_lag_168_time]),
        }

        feature_row = pd.DataFrame(
            [features],
            columns=model_features,
        )

        direct_prediction = max(
            0.0,
            float(direct_model.predict(feature_row)[0])
        )

        recursive_prediction = max(
            0.0,
            float(
                recursive_predictions.iloc[hours_ahead - 1][
                    "predicted_pickups"
                ]
            )
        )

        actual = float(history_lookup[target])

        previous_week = float(
            history_lookup[target - pd.Timedelta(hours=168)]
        )

        all_results.append({
            "origin_timestamp": origin,
            "forecast_timestamp": target,
            "hours_ahead": hours_ahead,
            "actual": actual,
            "direct_prediction": direct_prediction,
            "recursive_prediction": recursive_prediction,
            "previous_week_prediction": previous_week,
        })

    if index % 20 == 0 or index == len(valid_origins):
        print(f"Processed {index}/{len(valid_origins)} origins")


results = pd.DataFrame(all_results)

print("\nCalculating final test metrics...")

methods = {
    "Direct XGBoost": "direct_prediction",
    "Recursive XGBoost": "recursive_prediction",
    "Previous Week Baseline": "previous_week_prediction",
}

overall_metrics = []

for method, column in methods.items():
    metrics = calculate_metrics(
        results["actual"],
        results[column],
    )

    overall_metrics.append({
        "method": method,
        "samples": len(results),
        **metrics,
    })

overall = pd.DataFrame(overall_metrics)

print("\n" + "=" * 75)
print("FINAL UNTOUCHED TEST COMPARISON")
print("=" * 75)
print(overall.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

horizon_rows = []

for horizon in range(1, 25):
    subset = results[results["hours_ahead"] == horizon]

    row = {"hours_ahead": horizon}

    for method, column in methods.items():
        row[method] = calculate_metrics(
            subset["actual"],
            subset[column],
        )["MAE"]

    horizon_rows.append(row)

horizon_metrics = pd.DataFrame(horizon_rows)

print("\nPER-HORIZON MAE")
print("-" * 75)
print(horizon_metrics.to_string(
    index=False,
    float_format=lambda x: f"{x:,.2f}"
))

direct_mae = overall.loc[
    overall["method"] == "Direct XGBoost", "MAE"
].iloc[0]

recursive_mae = overall.loc[
    overall["method"] == "Recursive XGBoost", "MAE"
].iloc[0]

baseline_mae = overall.loc[
    overall["method"] == "Previous Week Baseline", "MAE"
].iloc[0]

print("\nFINAL COMPARISON")
print("-" * 75)
print(f"Direct vs baseline MAE improvement: {(1 - direct_mae / baseline_mae) * 100:.2f}%")
print(f"Recursive vs baseline MAE improvement: {(1 - recursive_mae / baseline_mae) * 100:.2f}%")

if direct_mae < recursive_mae:
    print("Best overall approach: DIRECT XGBOOST")
else:
    print("Best overall approach: RECURSIVE XGBOOST")

predictions_path = OUTPUT_DIR / "direct_vs_recursive_test_predictions.csv"
metrics_path = OUTPUT_DIR / "direct_vs_recursive_test_metrics.csv"
horizon_path = OUTPUT_DIR / "direct_vs_recursive_test_horizon_metrics.csv"

results.to_csv(predictions_path, index=False)
overall.to_csv(metrics_path, index=False)
horizon_metrics.to_csv(horizon_path, index=False)

print("\nSaved outputs:")
print(predictions_path)
print(metrics_path)
print(horizon_path)

print("\nIndependent test evaluation complete.")
