from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/ml/processed/direct_forecasting_samples.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports/ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "moveiq_direct_demand_forecaster.joblib"
METRICS_FILE = REPORT_DIR / "direct_model_validation_metrics.json"
HORIZON_FILE = REPORT_DIR / "direct_model_horizon_metrics.csv"

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

print("Loading direct forecasting dataset...")

data = pd.read_parquet(DATA_FILE)

train = data[data["dataset_split"] == "train"].copy()
validation = data[data["dataset_split"] == "validation"].copy()

X_train = train[FEATURES]
y_train = train["actual_demand"]

X_validation = validation[FEATURES]
y_validation = validation["actual_demand"]

print(f"Training samples   : {len(train):,}")
print(f"Validation samples : {len(validation):,}")
print(f"Features           : {len(FEATURES)}")

print("\nTraining Direct XGBoost model...")

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

start_time = time.time()

model.fit(X_train, y_train)

training_seconds = time.time() - start_time

print(f"Training completed in {training_seconds:.2f} seconds.")

print("\nEvaluating validation predictions...")

predictions = np.maximum(0, model.predict(X_validation))

overall_mae = mean_absolute_error(y_validation, predictions)
overall_rmse = np.sqrt(mean_squared_error(y_validation, predictions))
mean_error = float(np.mean(y_validation.to_numpy() - predictions))

print("\n" + "=" * 70)
print("DIRECT XGBOOST — VALIDATION RESULTS")
print("=" * 70)

print(f"MAE       : {overall_mae:,.2f}")
print(f"RMSE      : {overall_rmse:,.2f}")
print(f"Mean error: {mean_error:,.2f}")
print(f"Training  : {training_seconds:.2f} seconds")

validation["predicted_demand"] = predictions
validation["absolute_error"] = (
    validation["actual_demand"] - validation["predicted_demand"]
).abs()

horizon_metrics = (
    validation.groupby("hours_ahead")
    .apply(
        lambda group: pd.Series({
            "samples": len(group),
            "mae": mean_absolute_error(
                group["actual_demand"],
                group["predicted_demand"],
            ),
            "rmse": np.sqrt(mean_squared_error(
                group["actual_demand"],
                group["predicted_demand"],
            )),
        }),
        include_groups=False,
    )
    .reset_index()
)

print("\nPER-HORIZON VALIDATION PERFORMANCE")
print("-" * 70)
print(horizon_metrics.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

horizon_metrics.to_csv(HORIZON_FILE, index=False)

metrics = {
    "model_name": "MoveIQ Direct 24-Hour Demand Forecaster",
    "algorithm": "XGBoost Regressor",
    "forecasting_method": "Direct multi-horizon",
    "forecast_horizons": 24,
    "training_samples": int(len(train)),
    "validation_samples": int(len(validation)),
    "features": FEATURES,
    "validation_mae": float(overall_mae),
    "validation_rmse": float(overall_rmse),
    "validation_mean_error": mean_error,
    "training_seconds": training_seconds,
    "parameters": model.get_params(),
}

joblib.dump(model, MODEL_FILE)

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4, default=str)

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved validation metrics:")
print(METRICS_FILE)

print("\nSaved horizon metrics:")
print(HORIZON_FILE)

print("\nDirect model training complete.")
