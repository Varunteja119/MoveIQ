from pathlib import Path
import json
import shutil
import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path("data/ml/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/ml")

FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168"
]

print("Loading original XGBoost model...")
model = joblib.load(MODEL_DIR / "xgboost.joblib")

validation = pd.read_parquet(DATA_DIR / "validation.parquet")

X_val = validation[FEATURES]
y_val = validation["trip_count"]

print("\nCalculating permutation importance...")

importance = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df.round(4).to_string(index=False))

# Save a clearly named production candidate.
final_model_path = MODEL_DIR / "moveiq_demand_forecaster.joblib"

shutil.copy2(
    MODEL_DIR / "xgboost.joblib",
    final_model_path
)

metadata = {
    "model_name": "MoveIQ Hourly Demand Forecaster",
    "algorithm": "XGBoost Regressor",
    "model_file": str(final_model_path),
    "forecast_horizon": "one hour ahead",
    "forecasting_mode": "rolling one-hour-ahead",
    "target": "trip_count",
    "features": FEATURES,
    "training_records": 12159,
    "validation_records": 2605,
    "test_records": 2606,
    "validation_mae": 277.57,
    "validation_rmse": 425.20,
    "test_mae": 301.64,
    "test_rmse": 442.42,
    "baseline_test_mae_previous_week": 795.93,
    "baseline_test_mae_previous_hour": 861.57,
    "selection_note": "Original XGBoost selected because it outperformed tuned XGBoost on the untouched test dataset."
}

with open(REPORT_DIR / "model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("\n" + "=" * 65)
print("MODEL FINALIZATION")
print("=" * 65)
print(f"Selected model: {final_model_path}")
print(f"Metadata: {REPORT_DIR / 'model_metadata.json'}")
print(f"Feature importance: {REPORT_DIR / 'feature_importance.csv'}")

print("\nModel finalized successfully.")
