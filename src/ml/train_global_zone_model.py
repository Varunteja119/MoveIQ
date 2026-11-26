from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED = Path("data/ml/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/ml")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROCESSED / "zone_train.parquet"
VALIDATION_FILE = PROCESSED / "zone_validation.parquet"

MODEL_FILE = MODEL_DIR / "moveiq_global_zone_demand_forecaster.joblib"
METADATA_FILE = REPORT_DIR / "global_zone_model_metadata.json"
IMPORTANCE_FILE = REPORT_DIR / "global_zone_feature_importance.csv"

CATEGORICAL_FEATURES = [
    "zone_id",
    "borough",
    "service_zone",
]

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "trip_count"

print("=" * 75)
print("MOVEIQ GLOBAL ZONE DEMAND MODEL")
print("=" * 75)

print("\nLoading training and validation datasets...")

train = pd.read_parquet(TRAIN_FILE, columns=FEATURES + [TARGET])
validation = pd.read_parquet(VALIDATION_FILE, columns=FEATURES + [TARGET])

print(f"Training records   : {len(train):,}")
print(f"Validation records : {len(validation):,}")
print(f"Features           : {len(FEATURES)}")

print("\nPreparing categorical features...")

for column in CATEGORICAL_FEATURES:
    train[column] = train[column].astype("category")

    categories = train[column].cat.categories

    validation[column] = pd.Categorical(
        validation[column],
        categories=categories
    )

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

print("\nCalculating previous-week baseline...")

baseline_predictions = validation["lag_168"].to_numpy()
actual = y_validation.to_numpy()

baseline_mae = mean_absolute_error(actual, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_predictions))
baseline_bias = float(np.mean(baseline_predictions - actual))

print(f"Previous-week MAE  : {baseline_mae:,.4f}")
print(f"Previous-week RMSE : {baseline_rmse:,.4f}")
print(f"Previous-week bias : {baseline_bias:,.4f}")

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
    random_state=42,
    n_jobs=-1
)

print("\nTraining global model...")
print("This model learns across all active zones.")

start = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=50
)

training_seconds = time.time() - start

print(f"\nTraining completed in {training_seconds:.2f} seconds.")

print("\nEvaluating validation predictions...")

predictions = model.predict(X_validation)
predictions = np.maximum(predictions, 0)

model_mae = mean_absolute_error(actual, predictions)
model_rmse = np.sqrt(mean_squared_error(actual, predictions))
model_bias = float(np.mean(predictions - actual))

mae_improvement = (
    (baseline_mae - model_mae) / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - model_rmse) / baseline_rmse
) * 100

print("\n" + "=" * 75)
print("GLOBAL ZONE MODEL - VALIDATION RESULTS")
print("=" * 75)

print(f"{'Metric':<25}{'Previous-week':>20}{'Global XGBoost':>20}")
print("-" * 65)
print(f"{'MAE':<25}{baseline_mae:>20,.4f}{model_mae:>20,.4f}")
print(f"{'RMSE':<25}{baseline_rmse:>20,.4f}{model_rmse:>20,.4f}")
print(f"{'Mean prediction error':<25}{baseline_bias:>20,.4f}{model_bias:>20,.4f}")

print(f"\nMAE improvement : {mae_improvement:.2f}%")
print(f"RMSE improvement: {rmse_improvement:.2f}%")

print("\nSaving model...")

joblib.dump(model, MODEL_FILE)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(IMPORTANCE_FILE, index=False)

metadata = {
    "model_name": "MoveIQ Global Zone Demand Forecaster",
    "algorithm": "XGBoost Regressor",
    "forecasting_method": "Global zone-level one-step prediction",
    "model_file": str(MODEL_FILE),
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "training": {
        "training_samples": len(train),
        "validation_samples": len(validation),
        "active_zones": int(train["zone_id"].nunique()),
        "training_seconds": training_seconds,
        "parameters": model.get_params()
    },
    "validation_performance": {
        "previous_week_baseline_mae": float(baseline_mae),
        "previous_week_baseline_rmse": float(baseline_rmse),
        "previous_week_baseline_bias": baseline_bias,
        "global_xgboost_mae": float(model_mae),
        "global_xgboost_rmse": float(model_rmse),
        "global_xgboost_bias": model_bias,
        "mae_improvement_percent": float(mae_improvement),
        "rmse_improvement_percent": float(rmse_improvement)
    },
    "evaluation_notes": [
        "Validation data used for model evaluation.",
        "Test data was not loaded or used.",
        "Zone identity and geography are categorical features.",
        "Predictions are clipped to nonnegative values for evaluation.",
        "This is an initial global model, not yet the final selected model."
    ]
}

with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, default=str)

print("\nTop feature importance:")
print(importance.head(10).to_string(index=False))

print("\nSaved files:")
print(f"Model       : {MODEL_FILE}")
print(f"Metadata    : {METADATA_FILE}")
print(f"Importance  : {IMPORTANCE_FILE}")

print("\nSUCCESS: Global zone model trained and evaluated.")
print("The untouched test dataset remains unused.")
