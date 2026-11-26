import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "ml" / "processed"
REPORT_DIR = ROOT / "reports" / "ml"

CITY_MODEL = MODEL_DIR / "moveiq_direct_demand_forecaster_2024_2026_candidate.joblib"
ZONE_MODEL = MODEL_DIR / "moveiq_zone_direct_demand_forecaster_2024_2026_candidate.joblib"

CITY_TEST = DATA_DIR / "direct_test_2024_2026.parquet"
ZONE_TEST = DATA_DIR / "direct_zone_test_2024_2026.parquet"

TARGET_CANDIDATES = [
    "target",
    "target_demand",
    "target_pickups",
    "actual_demand",
    "actual_pickups",
    "target_trip_count",
    "trip_count",
    "pickup_count",
]

ZONE_CATEGORICAL = {"zone_id", "borough", "service_zone"}


def identify_target(df, features):
    for name in TARGET_CANDIDATES:
        if name in df.columns and name not in features:
            return name

    excluded = set(features) | {
        "origin_hour", "target_hour", "hours_ahead",
        "zone_id", "borough", "service_zone",
        "zone", "pickup_hour", "origin_timestamp",
        "target_timestamp",
    }

    candidates = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Could not identify the target column. "
        f"Remaining numeric candidates: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def get_features(model, fallback):
    try:
        names = model.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass

    try:
        names = model.feature_names_in_
        if names is not None:
            return list(names)
    except Exception:
        pass

    return fallback


def prepare_features(df, features, zone=False):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = df[features].copy()

    if zone:
        for col in ZONE_CATEGORICAL.intersection(features):
            X[col] = X[col].astype("category")

    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = pd.to_numeric(X[col], downcast="integer")

    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = pd.to_numeric(X[col], downcast="float")

    return X


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "bias": float(np.mean(predicted - actual)),
    }


def evaluate_city():
    print("\n" + "=" * 78)
    print("CITYWIDE DIRECT MODEL — 2026 HELD-OUT TEST")
    print("=" * 78)

    model = joblib.load(CITY_MODEL)
    df = pd.read_parquet(CITY_TEST)

    fallback = [
        "target_hour", "target_day_of_week", "target_is_weekend",
        "target_month", "hours_ahead", "origin_demand",
        "origin_lag_24", "origin_lag_168",
        "rolling_mean_24", "rolling_mean_168",
        "target_lag_24", "target_lag_168",
    ]

    features = get_features(model, fallback)
    target_col = identify_target(df, features)

    print(f"Test rows       : {len(df):,}")
    print(f"Target column   : {target_col}")
    print(f"Feature count   : {len(features)}")

    actual = df[target_col].to_numpy(dtype=np.float64)
    baseline = df["target_lag_168"].to_numpy(dtype=np.float64)

    X = prepare_features(df, features)
    started = time.time()
    predicted = model.predict(X).astype(np.float64)

    model_metrics = calculate_metrics(actual, predicted)
    baseline_metrics = calculate_metrics(actual, baseline)

    print("\nCITYWIDE TEST RESULTS")
    print(f"Model MAE       : {model_metrics['mae']:.4f}")
    print(f"Model RMSE      : {model_metrics['rmse']:.4f}")
    print(f"Model bias      : {model_metrics['bias']:.4f}")
    print(f"Weekly MAE      : {baseline_metrics['mae']:.4f}")
    print(f"Weekly RMSE     : {baseline_metrics['rmse']:.4f}")
    print(
        f"MAE improvement : "
        f"{(1 - model_metrics['mae'] / baseline_metrics['mae']) * 100:.2f}%"
    )
    print(f"Evaluation time : {(time.time() - started) / 60:.2f} minutes")

    results = df[["hours_ahead"]].copy()
    results["actual"] = actual
    results["prediction"] = predicted
    results["baseline"] = baseline

    horizon_rows = []
    for horizon, group in results.groupby("hours_ahead", sort=True):
        m = calculate_metrics(group["actual"], group["prediction"])
        b = calculate_metrics(group["actual"], group["baseline"])
        horizon_rows.append({
            "horizon": int(horizon),
            **{f"model_{k}": v for k, v in m.items()},
            "baseline_mae": b["mae"],
            "baseline_rmse": b["rmse"],
            "mae_improvement_pct": (1 - m["mae"] / b["mae"]) * 100,
        })

    horizon_df = pd.DataFrame(horizon_rows)
    horizon_path = REPORT_DIR / "direct_candidate_2026_test_by_horizon.csv"
    horizon_df.to_csv(horizon_path, index=False)

    metadata = {
        "evaluation": "2026 held-out test",
        "test_rows": int(len(df)),
        "target_column": target_col,
        "features": features,
        "model": model_metrics,
        "weekly_baseline": baseline_metrics,
        "mae_improvement_pct": float(
            (1 - model_metrics["mae"] / baseline_metrics["mae"]) * 100
        ),
        "horizon_report": str(horizon_path.relative_to(ROOT)),
    }

    metadata_path = REPORT_DIR / "direct_candidate_2026_test_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved horizon report: {horizon_path}")
    print(f"Saved metadata      : {metadata_path}")

    del model, df, X, results, predicted, actual, baseline
    gc.collect()


def evaluate_zone():
    print("\n" + "=" * 78)
    print("GLOBAL ZONE DIRECT MODEL — 2026 HELD-OUT TEST")
    print("=" * 78)

    model = joblib.load(ZONE_MODEL)

    print("Loading zone test dataset...")
    df = pd.read_parquet(ZONE_TEST)
    print(f"Test rows       : {len(df):,}")
    print(f"Test zones      : {df['zone_id'].nunique():,}")

    fallback = [
        "target_hour", "target_day_of_week", "target_is_weekend",
        "target_month", "hours_ahead", "origin_demand",
        "origin_lag_24", "origin_lag_168",
        "rolling_mean_24", "rolling_mean_168",
        "target_lag_24", "target_lag_168",
        "zone_id", "borough", "service_zone",
    ]

    features = get_features(model, fallback)
    target_col = identify_target(df, features)

    print(f"Target column   : {target_col}")
    print(f"Feature count   : {len(features)}")

    actual = df[target_col].to_numpy(dtype=np.float64)
    baseline = df["target_lag_168"].to_numpy(dtype=np.float64)

    X = prepare_features(df, features, zone=True)

    print("\nGenerating zone predictions...")
    started = time.time()
    predicted = model.predict(X).astype(np.float64)

    model_metrics = calculate_metrics(actual, predicted)
    baseline_metrics = calculate_metrics(actual, baseline)

    improvement_mae = (
        1 - model_metrics["mae"] / baseline_metrics["mae"]
    ) * 100
    improvement_rmse = (
        1 - model_metrics["rmse"] / baseline_metrics["rmse"]
    ) * 100

    print("\nZONE-LEVEL TEST RESULTS")
    print(f"Model MAE        : {model_metrics['mae']:.4f}")
    print(f"Model RMSE       : {model_metrics['rmse']:.4f}")
    print(f"Model bias       : {model_metrics['bias']:.4f}")
    print(f"Weekly MAE       : {baseline_metrics['mae']:.4f}")
    print(f"Weekly RMSE      : {baseline_metrics['rmse']:.4f}")
    print(f"MAE improvement  : {improvement_mae:.2f}%")
    print(f"RMSE improvement : {improvement_rmse:.2f}%")
    print(f"Evaluation time  : {(time.time() - started) / 60:.2f} minutes")

    results = df[["hours_ahead", "borough"]].copy()
    results["actual"] = actual
    results["prediction"] = predicted
    results["baseline"] = baseline

    horizon_rows = []
    for horizon, group in results.groupby("hours_ahead", sort=True):
        m = calculate_metrics(group["actual"], group["prediction"])
        b = calculate_metrics(group["actual"], group["baseline"])
        horizon_rows.append({
            "horizon": int(horizon),
            **{f"model_{k}": v for k, v in m.items()},
            "baseline_mae": b["mae"],
            "baseline_rmse": b["rmse"],
            "mae_improvement_pct": (1 - m["mae"] / b["mae"]) * 100,
            "rmse_improvement_pct": (1 - m["rmse"] / b["rmse"]) * 100,
        })

    horizon_df = pd.DataFrame(horizon_rows)
    horizon_path = REPORT_DIR / "zone_direct_candidate_2026_test_by_horizon.csv"
    horizon_df.to_csv(horizon_path, index=False)

    borough_rows = []
    for borough, group in results.groupby("borough", observed=True):
        m = calculate_metrics(group["actual"], group["prediction"])
        b = calculate_metrics(group["actual"], group["baseline"])
        borough_rows.append({
            "borough": str(borough),
            "rows": int(len(group)),
            **{f"model_{k}": v for k, v in m.items()},
            "baseline_mae": b["mae"],
            "baseline_rmse": b["rmse"],
            "mae_improvement_pct": (1 - m["mae"] / b["mae"]) * 100,
            "rmse_improvement_pct": (1 - m["rmse"] / b["rmse"]) * 100,
        })

    borough_df = pd.DataFrame(borough_rows)
    borough_path = REPORT_DIR / "zone_direct_candidate_2026_test_by_borough.csv"
    borough_df.to_csv(borough_path, index=False)

    metadata = {
        "evaluation": "2026 held-out test",
        "test_rows": int(len(df)),
        "test_zones": int(df["zone_id"].nunique()),
        "target_column": target_col,
        "features": features,
        "model": model_metrics,
        "weekly_baseline": baseline_metrics,
        "mae_improvement_pct": float(improvement_mae),
        "rmse_improvement_pct": float(improvement_rmse),
        "horizon_report": str(horizon_path.relative_to(ROOT)),
        "borough_report": str(borough_path.relative_to(ROOT)),
    }

    metadata_path = REPORT_DIR / "zone_direct_candidate_2026_test_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nBOROUGH RESULTS")
    print(
        borough_df[
            ["borough", "model_mae", "baseline_mae", "mae_improvement_pct"]
        ].to_string(index=False)
    )

    print(f"\nSaved horizon report : {horizon_path}")
    print(f"Saved borough report : {borough_path}")
    print(f"Saved metadata       : {metadata_path}")

    del model, df, X, results, predicted, actual, baseline
    gc.collect()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    required = [CITY_MODEL, ZONE_MODEL, CITY_TEST, ZONE_TEST]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Required files missing:\n" + "\n".join(missing))

    print("=" * 78)
    print("MOVEIQ — EXTENDED CANDIDATE 2026 TEST EVALUATION")
    print("=" * 78)
    print("No training or model updates will be performed.")
    print("Original benchmark models and reports will not be modified.")

    evaluate_city()
    evaluate_zone()

    print("\n" + "=" * 78)
    print("SUCCESS: BOTH CANDIDATES EVALUATED ON THE 2026 TEST SET")
    print("=" * 78)


if __name__ == "__main__":
    main()
