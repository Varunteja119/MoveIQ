from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

MODEL_FILE = Path("models/moveiq_global_zone_demand_forecaster.joblib")
VALIDATION_FILE = Path("data/ml/processed/zone_validation.parquet")
REPORT_DIR = Path("reports/ml")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "zone_id", "borough", "service_zone",
    "hour", "day_of_week", "is_weekend", "month",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_mean_168"
]

CATEGORICAL = ["zone_id", "borough", "service_zone"]

print("=" * 75)
print("GLOBAL ZONE MODEL - VALIDATION BREAKDOWN")
print("=" * 75)

print("\nLoading model and validation data...")

model = joblib.load(MODEL_FILE)
df = pd.read_parquet(VALIDATION_FILE)

for column in CATEGORICAL:
    df[column] = df[column].astype("category")

X = df[FEATURES]
actual = df["trip_count"].to_numpy()
baseline = df["lag_168"].to_numpy()

predictions = np.maximum(model.predict(X), 0)

df["global_prediction"] = predictions
df["baseline_prediction"] = baseline
df["global_absolute_error"] = np.abs(actual - predictions)
df["baseline_absolute_error"] = np.abs(actual - baseline)

def summarize(group):
    actual_values = group["trip_count"].to_numpy()
    global_pred = group["global_prediction"].to_numpy()
    baseline_pred = group["baseline_prediction"].to_numpy()

    global_mae = mean_absolute_error(actual_values, global_pred)
    baseline_mae = mean_absolute_error(actual_values, baseline_pred)

    return pd.Series({
        "records": len(group),
        "total_pickups": actual_values.sum(),
        "average_hourly_demand": actual_values.mean(),
        "global_mae": global_mae,
        "baseline_mae": baseline_mae,
        "mae_improvement_pct": (
            (baseline_mae - global_mae) / baseline_mae * 100
            if baseline_mae > 0 else np.nan
        )
    })

print("\n" + "=" * 75)
print("PERFORMANCE BY BOROUGH")
print("=" * 75)

borough_results = (
    df.groupby("borough", observed=True)
    .apply(summarize, include_groups=False)
    .sort_values("total_pickups", ascending=False)
)

print(borough_results.round(3).to_string())

print("\n" + "=" * 75)
print("PERFORMANCE BY ZONE")
print("=" * 75)

zone_results = (
    df.groupby(["zone_id", "borough", "service_zone"], observed=True)
    .apply(summarize, include_groups=False)
    .reset_index()
    .sort_values("total_pickups", ascending=False)
)

zone_results.to_csv(
    REPORT_DIR / "global_zone_validation_by_zone.csv",
    index=False
)

print("\nTOP 15 ZONES BY VALIDATION PICKUP VOLUME")
print(zone_results.head(15).round(3).to_string(index=False))

print("\n15 ZONES WITH HIGHEST GLOBAL MODEL MAE")
print(
    zone_results.sort_values("global_mae", ascending=False)
    .head(15)
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("ZONE-LEVEL MODEL VS BASELINE SUMMARY")
print("=" * 75)

better = (zone_results["mae_improvement_pct"] > 0).sum()
worse = (zone_results["mae_improvement_pct"] < 0).sum()
equal = (zone_results["mae_improvement_pct"] == 0).sum()

print(f"Zones where global model is better : {better}")
print(f"Zones where baseline is better     : {worse}")
print(f"Zones with equal MAE                : {equal}")

print("\nSaved:")
print(REPORT_DIR / "global_zone_validation_by_zone.csv")
print("\nTest data was not loaded or evaluated.")
