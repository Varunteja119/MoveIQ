import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "ml"

CITY_FEATURES = [
    "target_hour", "target_day_of_week", "target_is_weekend",
    "target_month", "hours_ahead", "origin_demand",
    "origin_lag_24", "origin_lag_168", "rolling_mean_24",
    "rolling_mean_168", "target_lag_24", "target_lag_168",
]
ZONE_FEATURES = CITY_FEATURES + ["zone_id", "borough", "service_zone"]
CAT_FEATURES = ["zone_id", "borough", "service_zone"]


def metrics(actual, pred):
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "Bias": float(np.mean(pred - actual)),
    }


def compare(label, test_file, target, feature_names, old_file, new_file):
    print("\n" + "=" * 76)
    print(label)
    print("=" * 76)

    df = pd.read_parquet(DATA / test_file)
    print(f"Test rows: {len(df):,}")

    actual = df[target].to_numpy(dtype=np.float64)
    baseline = df["target_lag_168"].to_numpy(dtype=np.float64)

    X = df[feature_names].copy()

    if "zone_id" in X.columns:
        for col in CAT_FEATURES:
            X[col] = X[col].astype("category")

    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = pd.to_numeric(X[col], downcast="integer")
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = pd.to_numeric(X[col], downcast="float")

    models = [
        ("Original 2024-25", MODELS / old_file),
        ("Extended 2024-26", MODELS / new_file),
    ]

    output = []
    predictions = {}

    for name, path in models:
        model = joblib.load(path)

        # XGBoost validates feature names and order against the saved model.
        model_features = model.get_booster().feature_names
        if model_features:
            X_model = X[model_features].copy()
        else:
            X_model = X.copy()

        pred = model.predict(X_model).astype(np.float64)
        predictions[name] = pred
        del X_model
        result = metrics(actual, pred)
        result["Model"] = name
        output.append(result)
        del model, pred
        gc.collect()

    baseline_metrics = metrics(actual, baseline)
    baseline_metrics["Model"] = "Previous-week baseline"
    output.append(baseline_metrics)

    result_df = pd.DataFrame(output)[["Model", "MAE", "RMSE", "Bias"]]
    print("\nOVERALL RESULTS")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    rows = []
    for horizon, indices in df.groupby("hours_ahead", sort=True).groups.items():
        idx = np.asarray(list(indices))
        y = actual[idx]
        base = baseline[idx]
        old = predictions["Original 2024-25"][idx]
        new = predictions["Extended 2024-26"][idx]

        old_m = metrics(y, old)
        new_m = metrics(y, new)
        base_m = metrics(y, base)

        rows.append({
            "horizon": int(horizon),
            "original_mae": old_m["MAE"],
            "extended_mae": new_m["MAE"],
            "baseline_mae": base_m["MAE"],
            "extended_vs_original_mae_pct": (
                (old_m["MAE"] - new_m["MAE"]) / old_m["MAE"] * 100
            ),
            "original_rmse": old_m["RMSE"],
            "extended_rmse": new_m["RMSE"],
            "baseline_rmse": base_m["RMSE"],
            "extended_vs_original_rmse_pct": (
                (old_m["RMSE"] - new_m["RMSE"]) / old_m["RMSE"] * 100
            ),
        })

    horizon_df = pd.DataFrame(rows)
    prefix = "city" if "citywide" in label.lower() else "zone"
    output_path = REPORTS / f"{prefix}_original_vs_extended_2026_test_by_horizon.csv"
    horizon_df.to_csv(output_path, index=False)

    print("\nHORIZON COMPARISON")
    print(horizon_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {output_path}")

    del df, X, actual, baseline, predictions
    gc.collect()


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    compare(
        "CITYWIDE — SAME 2026 TEST PERIOD",
        "direct_test_2024_2026.parquet",
        "actual_demand",
        CITY_FEATURES,
        "moveiq_direct_demand_forecaster.joblib",
        "moveiq_direct_demand_forecaster_2024_2026_candidate.joblib",
    )

    compare(
        "ZONE-LEVEL — SAME 2026 TEST PERIOD",
        "direct_zone_test_2024_2026.parquet",
        "actual_demand",
        ZONE_FEATURES,
        "moveiq_zone_direct_demand_forecaster.joblib",
        "moveiq_zone_direct_demand_forecaster_2024_2026_candidate.joblib",
    )

    print("\n" + "=" * 76)
    print("SAME-PERIOD MODEL COMPARISON COMPLETE")
    print("No models were trained or modified.")
    print("=" * 76)


if __name__ == "__main__":
    main()
