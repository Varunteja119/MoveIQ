from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/ml/processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports/ml"

MODEL_FILE = MODEL_DIR / "moveiq_zone_direct_demand_forecaster.joblib"
TRAIN_FILE = DATA_DIR / "zone_direct_train.parquet"
TEST_FILE = DATA_DIR / "zone_direct_test.parquet"

METRICS_FILE = REPORT_DIR / "zone_direct_test_metrics.json"
HORIZON_FILE = REPORT_DIR / "zone_direct_test_by_horizon.csv"
BOROUGH_FILE = REPORT_DIR / "zone_direct_test_by_borough.csv"
ZONE_FILE = REPORT_DIR / "zone_direct_test_by_zone.csv"

CATEGORICAL_FEATURES = [
    "zone_id",
    "borough",
    "service_zone",
]

NUMERIC_FEATURES = [
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

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "actual_demand"
COLUMNS = FEATURES + [TARGET]

INTEGER_FEATURES = [
    "target_hour",
    "target_day_of_week",
    "target_is_weekend",
    "target_month",
    "hours_ahead",
    "origin_demand",
    "origin_lag_24",
    "origin_lag_168",
    "target_lag_24",
    "target_lag_168",
]


print("=" * 75)
print("MOVEIQ ZONE DIRECT MODEL — UNTOUCHED TEST EVALUATION")
print("=" * 75)

print("\nLoading saved model...")
model = joblib.load(MODEL_FILE)

print(f"XGBoost version: {xgb.__version__}")
print(f"Model features : {len(FEATURES)}")

print("\nRecovering original training categories...")

category_source = pd.read_parquet(
    TRAIN_FILE,
    columns=CATEGORICAL_FEATURES
)

category_definitions = {}

for column in CATEGORICAL_FEATURES:
    category_source[column] = category_source[column].astype("category")
    category_definitions[column] = (
        category_source[column].cat.categories
    )

    print(
        f"{column}: "
        f"{len(category_definitions[column])} categories"
    )

del category_source

print("\nLoading untouched test dataset...")
start = time.time()

test = pd.read_parquet(TEST_FILE, columns=COLUMNS)

print(f"Test samples: {len(test):,}")
print(f"Loading time: {time.time() - start:.2f} seconds")

print("\nPreparing features...")

for column in CATEGORICAL_FEATURES:
    test[column] = pd.Categorical(
        test[column],
        categories=category_definitions[column]
    )

for column in INTEGER_FEATURES:
    test[column] = pd.to_numeric(
        test[column],
        downcast="integer"
    )

for column in ["rolling_mean_24", "rolling_mean_168"]:
    test[column] = test[column].astype("float32")

test[TARGET] = test[TARGET].astype("float32")

if test[FEATURES].isnull().any().any():
    raise ValueError("Unexpected missing values in test features.")

if test[TARGET].isnull().any():
    raise ValueError("Unexpected missing target values.")

print(f"Test DataFrame memory: {test.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
print(f"Zones: {test['zone_id'].nunique()}")

print("\nGenerating predictions...")
prediction_start = time.time()

X_test = test[FEATURES]
actual = test[TARGET].to_numpy(dtype=np.float32, copy=False)

predictions = np.maximum(
    0,
    model.predict(X_test)
).astype(np.float32)

prediction_seconds = time.time() - prediction_start

baseline = test["target_lag_168"].to_numpy(
    dtype=np.float32,
    copy=False
)

print(f"Prediction time: {prediction_seconds:.2f} seconds")

print("\nCalculating overall metrics...")

model_mae = float(mean_absolute_error(actual, predictions))
model_rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
model_bias = float(np.mean(predictions - actual))

baseline_mae = float(mean_absolute_error(actual, baseline))
baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline)))
baseline_bias = float(np.mean(baseline - actual))

mae_improvement = (
    (baseline_mae - model_mae) / baseline_mae * 100
)

rmse_improvement = (
    (baseline_rmse - model_rmse) / baseline_rmse * 100
)

print("\n" + "=" * 75)
print("OVERALL TEST RESULTS")
print("=" * 75)

print(f"Previous-week MAE : {baseline_mae:.4f}")
print(f"Model MAE         : {model_mae:.4f}")
print(f"MAE improvement  : {mae_improvement:.2f}%")

print("-" * 75)

print(f"Previous-week RMSE: {baseline_rmse:.4f}")
print(f"Model RMSE        : {model_rmse:.4f}")
print(f"RMSE improvement : {rmse_improvement:.2f}%")

print("-" * 75)

print(f"Previous-week bias: {baseline_bias:.4f}")
print(f"Model bias        : {model_bias:.4f}")

print("\nEvaluating forecast horizons...")

test["predicted_demand"] = predictions
test["baseline_prediction"] = baseline

horizon_results = []

for horizon in range(1, 25):
    group = test[test["hours_ahead"] == horizon]

    y = group[TARGET].to_numpy()
    p = group["predicted_demand"].to_numpy()
    b = group["baseline_prediction"].to_numpy()

    horizon_results.append({
        "hours_ahead": horizon,
        "samples": len(group),
        "model_mae": float(mean_absolute_error(y, p)),
        "baseline_mae": float(mean_absolute_error(y, b)),
        "model_rmse": float(np.sqrt(mean_squared_error(y, p))),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y, b))),
        "model_bias": float(np.mean(p - y)),
    })

horizon_df = pd.DataFrame(horizon_results)
horizon_df.to_csv(HORIZON_FILE, index=False)

print(horizon_df.to_string(index=False))

print("\nEvaluating borough performance...")

borough_results = []

for borough, group in test.groupby("borough", observed=True):
    y = group[TARGET].to_numpy()
    p = group["predicted_demand"].to_numpy()
    b = group["baseline_prediction"].to_numpy()

    borough_results.append({
        "borough": str(borough),
        "samples": len(group),
        "actual_demand": float(y.sum()),
        "model_mae": float(mean_absolute_error(y, p)),
        "baseline_mae": float(mean_absolute_error(y, b)),
        "model_rmse": float(np.sqrt(mean_squared_error(y, p))),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y, b))),
        "mae_improvement_percent": float(
            (mean_absolute_error(y, b) - mean_absolute_error(y, p))
            / mean_absolute_error(y, b) * 100
        ),
    })

borough_df = pd.DataFrame(borough_results)
borough_df.to_csv(BOROUGH_FILE, index=False)

print(borough_df.to_string(index=False))

print("\nEvaluating individual zones...")

zone_results = []

for zone_id, group in test.groupby("zone_id", observed=True):
    y = group[TARGET].to_numpy()
    p = group["predicted_demand"].to_numpy()
    b = group["baseline_prediction"].to_numpy()

    model_zone_mae = float(mean_absolute_error(y, p))
    baseline_zone_mae = float(mean_absolute_error(y, b))

    zone_results.append({
        "zone_id": int(zone_id),
        "borough": str(group["borough"].iloc[0]),
        "samples": len(group),
        "actual_demand": float(y.sum()),
        "model_mae": model_zone_mae,
        "baseline_mae": baseline_zone_mae,
        "mae_improvement_percent": (
            (baseline_zone_mae - model_zone_mae)
            / baseline_zone_mae * 100
            if baseline_zone_mae > 0 else None
        ),
    })

zone_df = pd.DataFrame(zone_results)
zone_df.to_csv(ZONE_FILE, index=False)

better_zones = int(
    (zone_df["model_mae"] < zone_df["baseline_mae"]).sum()
)

worse_zones = int(
    (zone_df["model_mae"] > zone_df["baseline_mae"]).sum()
)

print(f"\nZones evaluated       : {len(zone_df)}")
print(f"Model better          : {better_zones}")
print(f"Baseline better       : {worse_zones}")

print("\nTop 10 zones by actual demand:")
print(
    zone_df.sort_values("actual_demand", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nZones where baseline performs better:")
print(
    zone_df[zone_df["model_mae"] > zone_df["baseline_mae"]]
    .sort_values("actual_demand", ascending=False)
    .head(20)
    .to_string(index=False)
)

metadata = {
    "model": str(MODEL_FILE),
    "evaluation_split": "test",
    "test_samples": len(test),
    "zones_evaluated": len(zone_df),
    "forecast_horizons": list(range(1, 25)),
    "test_data_used_for_training": False,
    "baseline": {
        "name": "Previous-week target lag",
        "mae": baseline_mae,
        "rmse": baseline_rmse,
        "bias": baseline_bias,
    },
    "model": {
        "mae": model_mae,
        "rmse": model_rmse,
        "bias": model_bias,
        "mae_improvement_percent": mae_improvement,
        "rmse_improvement_percent": rmse_improvement,
    },
    "zones_model_better": better_zones,
    "zones_baseline_better": worse_zones,
    "prediction_seconds": prediction_seconds,
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nReports saved:")
print(METRICS_FILE)
print(HORIZON_FILE)
print(BOROUGH_FILE)
print(ZONE_FILE)

print("\n" + "=" * 75)
print("INDEPENDENT TEST EVALUATION COMPLETED")
print("=" * 75)
