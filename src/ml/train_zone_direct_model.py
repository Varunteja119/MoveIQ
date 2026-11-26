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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "zone_direct_train.parquet"
VALIDATION_FILE = DATA_DIR / "zone_direct_validation.parquet"

MODEL_FILE = MODEL_DIR / "moveiq_zone_direct_demand_forecaster.joblib"
METADATA_FILE = REPORT_DIR / "zone_direct_model_metadata.json"
IMPORTANCE_FILE = REPORT_DIR / "zone_direct_feature_importance.csv"
HORIZON_FILE = REPORT_DIR / "zone_direct_validation_by_horizon.csv"
BOROUGH_FILE = REPORT_DIR / "zone_direct_validation_by_borough.csv"

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
REQUIRED_COLUMNS = FEATURES + [TARGET]

print("=" * 75)
print("MOVEIQ GLOBAL ZONE DIRECT 24-HOUR FORECASTING MODEL")
print("=" * 75)

print("\nModel design:")
print("- One global model trained across all active zones.")
print("- One prediction per zone and forecast horizon.")
print("- Forecast horizons: 1 to 24 hours.")
print("- No recursive prediction feedback.")
print("- Test dataset will remain untouched.")

print("\nLoading training data...")
start_load = time.time()

train = pd.read_parquet(TRAIN_FILE, columns=REQUIRED_COLUMNS)

print(f"Training rows: {len(train):,}")
print(f"Loading time: {time.time() - start_load:.2f} seconds")

print("\nOptimizing training data types...")

for column in CATEGORICAL_FEATURES:
    train[column] = train[column].astype("category")

for column in NUMERIC_FEATURES:
    train[column] = pd.to_numeric(
        train[column], downcast="integer"
        if column in [
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
        else "float"
    )

train[TARGET] = train[TARGET].astype("float32")

train_memory = train.memory_usage(deep=True).sum() / (1024 ** 3)
print(f"Training DataFrame memory: {train_memory:.2f} GB")

print("\nLoading validation data...")
validation = pd.read_parquet(
    VALIDATION_FILE,
    columns=REQUIRED_COLUMNS
)

print(f"Validation rows: {len(validation):,}")

print("\nApplying matching data types...")

for column in CATEGORICAL_FEATURES:
    validation[column] = pd.Categorical(
        validation[column],
        categories=train[column].cat.categories
    )

for column in NUMERIC_FEATURES:
    validation[column] = pd.to_numeric(
        validation[column], downcast="integer"
        if column in [
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
        else "float"
    )

validation[TARGET] = validation[TARGET].astype("float32")

validation_memory = validation.memory_usage(deep=True).sum() / (1024 ** 3)
print(f"Validation DataFrame memory: {validation_memory:.2f} GB")

X_train = train[FEATURES]
y_train = train[TARGET].to_numpy(dtype=np.float32, copy=False)

X_validation = validation[FEATURES]
y_validation = validation[TARGET].to_numpy(dtype=np.float32, copy=False)

print("\nDataset checks:")
print(f"Training samples   : {len(X_train):,}")
print(f"Validation samples : {len(X_validation):,}")
print(f"Features           : {len(FEATURES)}")
print(f"Training zones     : {train['zone_id'].nunique()}")
print(f"Forecast horizons  : {sorted(train['hours_ahead'].unique().tolist())}")

print("\nCalculating previous-week baseline...")

baseline_predictions = validation["target_lag_168"].to_numpy(
    dtype=np.float32
)

baseline_mae = float(
    mean_absolute_error(y_validation, baseline_predictions)
)
baseline_rmse = float(
    np.sqrt(mean_squared_error(y_validation, baseline_predictions))
)
baseline_bias = float(np.mean(baseline_predictions - y_validation))

print(f"Previous-week MAE  : {baseline_mae:.4f}")
print(f"Previous-week RMSE : {baseline_rmse:.4f}")
print(f"Previous-week bias : {baseline_bias:.4f}")

print("\nConfiguring XGBoost...")

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    max_bin=256,
    random_state=42,
    n_jobs=4,
)

print("\nStarting model training...")
training_start = time.time()

model.fit(
    X_train,
    y_train,
    verbose=50
)

training_seconds = time.time() - training_start

print(f"\nTraining completed in {training_seconds / 60:.2f} minutes.")

print("\nGenerating validation predictions...")
prediction_start = time.time()

predictions = np.maximum(
    0,
    model.predict(X_validation)
).astype(np.float32)

prediction_seconds = time.time() - prediction_start

model_mae = float(mean_absolute_error(y_validation, predictions))
model_rmse = float(np.sqrt(mean_squared_error(y_validation, predictions)))
model_bias = float(np.mean(predictions - y_validation))

mae_improvement = (
    (baseline_mae - model_mae) / baseline_mae * 100
    if baseline_mae else 0
)

rmse_improvement = (
    (baseline_rmse - model_rmse) / baseline_rmse * 100
    if baseline_rmse else 0
)

print("\n" + "=" * 75)
print("ZONE DIRECT MODEL — VALIDATION RESULTS")
print("=" * 75)

print(f"Previous-week MAE : {baseline_mae:.4f}")
print(f"Model MAE         : {model_mae:.4f}")
print(f"MAE improvement  : {mae_improvement:.2f}%")
print("-" * 75)
print(f"Previous-week RMSE: {baseline_rmse:.4f}")
print(f"Model RMSE        : {model_rmse:.4f}")
print(f"RMSE improvement : {rmse_improvement:.2f}%")
print("-" * 75)
print(f"Model bias        : {model_bias:.4f}")
print(f"Prediction time  : {prediction_seconds:.2f} seconds")

print("\nEvaluating each forecast horizon...")

horizon_results = []

for horizon in range(1, 25):
    mask = validation["hours_ahead"].to_numpy() == horizon

    actual_h = y_validation[mask]
    predicted_h = predictions[mask]
    baseline_h = baseline_predictions[mask]

    horizon_results.append({
        "hours_ahead": horizon,
        "samples": int(mask.sum()),
        "model_mae": float(mean_absolute_error(actual_h, predicted_h)),
        "baseline_mae": float(mean_absolute_error(actual_h, baseline_h)),
        "model_rmse": float(
            np.sqrt(mean_squared_error(actual_h, predicted_h))
        ),
        "baseline_rmse": float(
            np.sqrt(mean_squared_error(actual_h, baseline_h))
        ),
        "model_bias": float(np.mean(predicted_h - actual_h)),
    })

horizon_df = pd.DataFrame(horizon_results)
horizon_df.to_csv(HORIZON_FILE, index=False)

print(horizon_df.to_string(index=False))

print("\nEvaluating performance by borough...")

borough_results = []

for borough, group in validation.groupby("borough", observed=True):
    indices = group.index.to_numpy()
    positions = validation.index.get_indexer(indices)

    actual_b = y_validation[positions]
    predicted_b = predictions[positions]
    baseline_b = baseline_predictions[positions]

    borough_results.append({
        "borough": str(borough),
        "samples": len(positions),
        "actual_demand": float(actual_b.sum()),
        "model_mae": float(mean_absolute_error(actual_b, predicted_b)),
        "baseline_mae": float(mean_absolute_error(actual_b, baseline_b)),
        "model_rmse": float(
            np.sqrt(mean_squared_error(actual_b, predicted_b))
        ),
    })

borough_df = pd.DataFrame(borough_results)
borough_df.to_csv(BOROUGH_FILE, index=False)

print(borough_df.to_string(index=False))

print("\nSaving model...")
joblib.dump(model, MODEL_FILE, compress=3)

print(f"Model saved: {MODEL_FILE}")

print("\nSaving feature importance...")

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance_df.to_csv(IMPORTANCE_FILE, index=False)
print(importance_df.to_string(index=False))

metadata = {
    "model_name": "MoveIQ Global Zone Direct Demand Forecaster",
    "model_type": "XGBoost direct multi-horizon global zone model",
    "xgboost_version": xgb.__version__,
    "training_rows": len(train),
    "validation_rows": len(validation),
    "zones": int(train["zone_id"].nunique()),
    "forecast_horizons": list(range(1, 25)),
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "target": TARGET,
    "parameters": model.get_params(),
    "baseline": {
        "name": "Previous-week target lag",
        "mae": baseline_mae,
        "rmse": baseline_rmse,
        "bias": baseline_bias,
    },
    "validation": {
        "mae": model_mae,
        "rmse": model_rmse,
        "bias": model_bias,
        "mae_improvement_percent": mae_improvement,
        "rmse_improvement_percent": rmse_improvement,
    },
    "training_seconds": training_seconds,
    "prediction_seconds": prediction_seconds,
    "test_data_used": False,
}

with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, default=str)

print(f"Metadata saved: {METADATA_FILE}")

print("\n" + "=" * 75)
print("ZONE DIRECT MODEL TRAINING COMPLETED")
print("=" * 75)
print(f"Model       : {MODEL_FILE}")
print(f"Metadata    : {METADATA_FILE}")
print(f"Horizon CSV : {HORIZON_FILE}")
print(f"Borough CSV : {BOROUGH_FILE}")
print(f"Importance  : {IMPORTANCE_FILE}")
print("=" * 75)
