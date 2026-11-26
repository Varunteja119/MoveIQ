from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ml/processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports/ml"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 4,
}


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "bias": float(np.mean(predicted - actual)),
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def train_citywide():
    print("\n" + "=" * 78)
    print("CITYWIDE DIRECT 24-HOUR CANDIDATE")
    print("=" * 78)

    features = [
        "target_hour", "target_day_of_week", "target_is_weekend",
        "target_month", "hours_ahead", "origin_demand",
        "origin_lag_24", "origin_lag_168",
        "rolling_mean_24", "rolling_mean_168",
        "target_lag_24", "target_lag_168",
    ]

    train_path = DATA / "direct_train_2024_2026.parquet"
    val_path = DATA / "direct_validation_2024_2026.parquet"

    train = pd.read_parquet(train_path, columns=features + ["actual_demand"])
    val = pd.read_parquet(val_path, columns=features + ["actual_demand"])

    X_train = train[features]
    y_train = train["actual_demand"]
    X_val = val[features]
    y_val = val["actual_demand"]

    print(f"Training samples   : {len(train):,}")
    print(f"Validation samples : {len(val):,}")
    print(f"Features           : {len(features)}")

    model = xgb.XGBRegressor(**PARAMS)

    start = time.time()
    model.fit(X_train, y_train, verbose=50)
    training_seconds = time.time() - start

    predictions = np.maximum(0, model.predict(X_val))
    actual = y_val.to_numpy()

    model_metrics = metrics(actual, predictions)
    baseline_predictions = val["target_lag_168"].to_numpy()
    baseline_metrics = metrics(actual, baseline_predictions)

    print("\nCITYWIDE VALIDATION")
    print(f"Model MAE       : {model_metrics['mae']:,.4f}")
    print(f"Model RMSE      : {model_metrics['rmse']:,.4f}")
    print(f"Model bias      : {model_metrics['bias']:,.4f}")
    print(f"Weekly baseline : {baseline_metrics['mae']:,.4f} MAE")
    print(f"Training time   : {training_seconds / 60:.2f} minutes")

    horizon_rows = []
    for horizon in range(1, 25):
        mask = val["hours_ahead"].to_numpy() == horizon
        row = metrics(actual[mask], predictions[mask])
        row["hours_ahead"] = horizon
        row["samples"] = int(mask.sum())
        horizon_rows.append(row)

    pd.DataFrame(horizon_rows).to_csv(
        REPORTS / "direct_candidate_2024_2026_by_horizon.csv",
        index=False
    )

    model_path = MODELS / "moveiq_direct_demand_forecaster_2024_2026_candidate.joblib"
    joblib.dump(model, model_path)

    save_json(REPORTS / "direct_candidate_2024_2026_metadata.json", {
        "model_name": "MoveIQ Citywide Direct Demand Forecaster Candidate",
        "algorithm": "XGBoost",
        "xgboost_version": xgb.__version__,
        "training_rows": len(train),
        "validation_rows": len(val),
        "features": features,
        "parameters": model.get_params(),
        "training_seconds": training_seconds,
        "validation": model_metrics,
        "previous_week_baseline": baseline_metrics,
        "test_data_used": False,
        "model_file": str(model_path),
    })

    print(f"Candidate saved: {model_path}")

    del train, val, X_train, X_val, y_train, y_val
    del model, predictions, actual
    gc.collect()


def train_zone():
    print("\n" + "=" * 78)
    print("GLOBAL ZONE DIRECT 24-HOUR CANDIDATE")
    print("=" * 78)

    categorical = ["zone_id", "borough", "service_zone"]

    numeric = [
        "target_hour", "target_day_of_week", "target_is_weekend",
        "target_month", "hours_ahead", "origin_demand",
        "origin_lag_24", "origin_lag_168",
        "rolling_mean_24", "rolling_mean_168",
        "target_lag_24", "target_lag_168",
    ]

    features = categorical + numeric
    target = "actual_demand"
    columns = features + [target]

    train_path = DATA / "direct_zone_train_2024_2026.parquet"
    val_path = DATA / "direct_zone_validation_2024_2026.parquet"

    print("Loading zone training data...")
    start_load = time.time()
    train = pd.read_parquet(train_path, columns=columns)
    print(f"Training rows: {len(train):,}")
    print(f"Loading time: {time.time() - start_load:.1f} seconds")

    for column in categorical:
        train[column] = train[column].astype("category")

    integer_columns = {
        "target_hour", "target_day_of_week", "target_is_weekend",
        "target_month", "hours_ahead", "origin_demand",
        "origin_lag_24", "origin_lag_168",
        "target_lag_24", "target_lag_168",
    }

    for column in numeric:
        train[column] = pd.to_numeric(
            train[column],
            downcast="integer" if column in integer_columns else "float"
        )

    train[target] = train[target].astype("float32")

    print(
        f"Training DataFrame memory: "
        f"{train.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
    )

    print("Loading zone validation data...")
    val = pd.read_parquet(val_path, columns=columns)
    print(f"Validation rows: {len(val):,}")

    for column in categorical:
        val[column] = pd.Categorical(
            val[column],
            categories=train[column].cat.categories
        )

    for column in numeric:
        val[column] = pd.to_numeric(
            val[column],
            downcast="integer" if column in integer_columns else "float"
        )

    val[target] = val[target].astype("float32")

    X_train = train[features]
    y_train = train[target].to_numpy(dtype=np.float32, copy=False)
    X_val = val[features]
    y_val = val[target].to_numpy(dtype=np.float32, copy=False)

    print(f"Training zones: {train['zone_id'].nunique()}")
    print(f"Forecast horizons: {sorted(train['hours_ahead'].unique().tolist())}")

    baseline = val["target_lag_168"].to_numpy(dtype=np.float32)
    baseline_metrics = metrics(y_val, baseline)

    print("\nPrevious-week baseline:")
    print(f"MAE : {baseline_metrics['mae']:.4f}")
    print(f"RMSE: {baseline_metrics['rmse']:.4f}")

    params = {
        **PARAMS,
        "enable_categorical": True,
        "max_bin": 256,
    }

    model = xgb.XGBRegressor(**params)

    print("\nStarting global zone model training...")
    start = time.time()
    model.fit(X_train, y_train, verbose=50)
    training_seconds = time.time() - start

    print(f"Training completed in {training_seconds / 60:.2f} minutes.")

    predictions = np.maximum(0, model.predict(X_val)).astype(np.float32)
    model_metrics = metrics(y_val, predictions)

    mae_improvement = (
        (baseline_metrics["mae"] - model_metrics["mae"])
        / baseline_metrics["mae"] * 100
    )
    rmse_improvement = (
        (baseline_metrics["rmse"] - model_metrics["rmse"])
        / baseline_metrics["rmse"] * 100
    )

    print("\nZONE VALIDATION")
    print(f"Model MAE        : {model_metrics['mae']:.4f}")
    print(f"Model RMSE       : {model_metrics['rmse']:.4f}")
    print(f"Model bias       : {model_metrics['bias']:.4f}")
    print(f"MAE improvement : {mae_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")

    horizon_rows = []
    horizon_values = val["hours_ahead"].to_numpy()

    for horizon in range(1, 25):
        mask = horizon_values == horizon
        row = metrics(y_val[mask], predictions[mask])
        row["hours_ahead"] = horizon
        row["samples"] = int(mask.sum())
        row["baseline_mae"] = float(
            mean_absolute_error(y_val[mask], baseline[mask])
        )
        horizon_rows.append(row)

    pd.DataFrame(horizon_rows).to_csv(
        REPORTS / "zone_direct_candidate_2024_2026_by_horizon.csv",
        index=False
    )

    borough_rows = []
    borough_values = val["borough"].astype(str).to_numpy()

    for borough in sorted(set(borough_values)):
        mask = borough_values == borough
        row = metrics(y_val[mask], predictions[mask])
        row["borough"] = borough
        row["samples"] = int(mask.sum())
        borough_rows.append(row)

    pd.DataFrame(borough_rows).to_csv(
        REPORTS / "zone_direct_candidate_2024_2026_by_borough.csv",
        index=False
    )

    model_path = MODELS / "moveiq_zone_direct_demand_forecaster_2024_2026_candidate.joblib"
    joblib.dump(model, model_path, compress=3)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORTS / "zone_direct_candidate_2024_2026_feature_importance.csv",
        index=False
    )

    save_json(REPORTS / "zone_direct_candidate_2024_2026_metadata.json", {
        "model_name": "MoveIQ Global Zone Direct Demand Forecaster Candidate",
        "algorithm": "XGBoost",
        "xgboost_version": xgb.__version__,
        "training_rows": len(train),
        "validation_rows": len(val),
        "zones": int(train["zone_id"].nunique()),
        "forecast_horizons": list(range(1, 25)),
        "features": features,
        "parameters": model.get_params(),
        "training_seconds": training_seconds,
        "validation": model_metrics,
        "previous_week_baseline": baseline_metrics,
        "mae_improvement_percent": mae_improvement,
        "rmse_improvement_percent": rmse_improvement,
        "test_data_used": False,
        "model_file": str(model_path),
    })

    print(f"\nCandidate saved: {model_path}")
    print("Zone feature importance:")
    print(importance.head(10).to_string(index=False))


def main():
    print("=" * 78)
    print("MOVEIQ — EXTENDED CANDIDATE MODEL TRAINING")
    print("=" * 78)
    print("XGBoost:", xgb.__version__)
    print("The 2026 test datasets will not be loaded.")

    train_citywide()
    gc.collect()

    train_zone()
    gc.collect()

    print("\n" + "=" * 78)
    print("SUCCESS: BOTH CANDIDATE MODELS TRAINED")
    print("Existing benchmark models were not overwritten.")
    print("Test datasets remain untouched.")
    print("=" * 78)


if __name__ == "__main__":
    main()
