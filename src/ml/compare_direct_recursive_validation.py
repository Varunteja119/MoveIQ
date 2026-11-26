from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.ml.forecast_service import MoveIQForecastService

ROOT = Path(__file__).resolve().parents[2]

HISTORY_FILE = ROOT / "data/ml/processed/hourly_demand_2024_2025.parquet"
SAMPLES_FILE = ROOT / "data/ml/processed/direct_forecasting_samples.parquet"

DIRECT_MODEL_FILE = ROOT / "models/moveiq_direct_demand_forecaster.joblib"
RECURSIVE_MODEL_FILE = ROOT / "models/moveiq_demand_forecaster.joblib"

OUTPUT_FILE = ROOT / "reports/ml/direct_vs_recursive_validation.csv"
METRICS_FILE = ROOT / "reports/ml/direct_vs_recursive_validation_metrics.csv"

FEATURES = [
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

print("Loading models and validation samples...")

direct_model = joblib.load(DIRECT_MODEL_FILE)

recursive_service = MoveIQForecastService(
    history_path=HISTORY_FILE,
    model_path=RECURSIVE_MODEL_FILE,
)

samples = pd.read_parquet(SAMPLES_FILE)
history = pd.read_parquet(HISTORY_FILE)

history["pickup_hour"] = pd.to_datetime(history["pickup_hour"])
history = history.sort_values("pickup_hour").reset_index(drop=True)

validation = samples[samples["dataset_split"] == "validation"].copy()

# Use daily-spaced origins to keep this comparison consistent
# with the existing 24-hour recursive backtest methodology.
origins = (
    validation["origin_timestamp"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

origins = origins.iloc[::24].tolist()

print(f"Daily validation origins: {len(origins)}")

selected = validation[
    validation["origin_timestamp"].isin(origins)
].copy()

print(f"Direct forecast rows: {len(selected)}")

print("\nGenerating direct predictions...")

selected["direct_prediction"] = np.maximum(
    0,
    direct_model.predict(selected[FEATURES]),
)

print("Generating recursive predictions...")

recursive_results = []

for index, origin in enumerate(origins, start=1):
    observed_history = history[
        history["pickup_hour"] <= origin
    ].copy().reset_index(drop=True)

    if observed_history.empty or observed_history.iloc[-1]["pickup_hour"] != origin:
        raise ValueError(f"Origin not found in historical data: {origin}")

    recursive_service.history = observed_history

    forecast = recursive_service.forecast_next_24_hours()

    forecast = forecast.copy()
    forecast["origin_timestamp"] = origin

    recursive_results.append(forecast)

    if index % 20 == 0 or index == len(origins):
        print(f"Processed {index}/{len(origins)} origins")

recursive = pd.concat(recursive_results, ignore_index=True)

recursive = recursive.rename(columns={
    "forecast_timestamp": "target_timestamp",
    "predicted_pickups": "recursive_prediction",
})

comparison = selected.merge(
    recursive[
        [
            "origin_timestamp",
            "target_timestamp",
            "recursive_prediction",
        ]
    ],
    on=["origin_timestamp", "target_timestamp"],
    how="inner",
)

print("\nAdding previous-week baseline...")

lookup = dict(zip(history["pickup_hour"], history["trip_count"]))

comparison["previous_week_prediction"] = comparison[
    "target_timestamp"
].map(
    lambda timestamp: lookup.get(
        timestamp - pd.Timedelta(hours=168),
        np.nan,
    )
)

comparison = comparison.dropna(
    subset=["actual_demand", "previous_week_prediction"]
).copy()

comparison["recursive_prediction"] = comparison[
    "recursive_prediction"
].clip(lower=0)

comparison["direct_prediction"] = comparison[
    "direct_prediction"
].clip(lower=0)

comparison.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 75)
print("FAIR 24-HOUR VALIDATION COMPARISON")
print("=" * 75)

methods = {
    "Direct XGBoost": "direct_prediction",
    "Recursive XGBoost": "recursive_prediction",
    "Previous Week Baseline": "previous_week_prediction",
}

rows = []

for method, column in methods.items():
    mae = mean_absolute_error(
        comparison["actual_demand"],
        comparison[column],
    )

    rmse = np.sqrt(mean_squared_error(
        comparison["actual_demand"],
        comparison[column],
    ))

    rows.append({
        "method": method,
        "samples": len(comparison),
        "MAE": mae,
        "RMSE": rmse,
    })

metrics = pd.DataFrame(rows)
print(metrics.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

metrics.to_csv(METRICS_FILE, index=False)

print("\nPER-HORIZON MAE")
print("-" * 75)

horizon_rows = []

for horizon, group in comparison.groupby("hours_ahead"):
    row = {"hours_ahead": horizon}

    for method, column in methods.items():
        row[method] = mean_absolute_error(
            group["actual_demand"],
            group[column],
        )

    horizon_rows.append(row)

horizon_metrics = pd.DataFrame(horizon_rows)
print(horizon_metrics.to_string(
    index=False,
    float_format=lambda x: f"{x:,.2f}",
))

print(f"\nSaved predictions: {OUTPUT_FILE}")
print(f"Saved metrics: {METRICS_FILE}")

print("\nValidation comparison complete.")
