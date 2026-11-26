import json
import math
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "moveiq_direct_demand_forecaster.joblib"
VALIDATION_PATH = ROOT / "reports" / "ml" / "direct_model_validation_metrics.json"
TEST_PATH = ROOT / "reports" / "ml" / "direct_vs_recursive_test_metrics.csv"
HORIZON_PATH = ROOT / "reports" / "ml" / "direct_vs_recursive_test_horizon_metrics.csv"
OUTPUT_PATH = ROOT / "reports" / "ml" / "direct_model_metadata.json"

# Load existing artifacts
model = joblib.load(MODEL_PATH)

with open(VALIDATION_PATH, "r", encoding="utf-8") as file:
    validation = json.load(file)

test_metrics = pd.read_csv(TEST_PATH)
horizon_metrics = pd.read_csv(HORIZON_PATH)

# Extract Direct XGBoost test results
direct_test = test_metrics.loc[
    test_metrics["method"] == "Direct XGBoost"
].iloc[0]

baseline_test = test_metrics.loc[
    test_metrics["method"] == "Previous Week Baseline"
].iloc[0]

recursive_test = test_metrics.loc[
    test_metrics["method"] == "Recursive XGBoost"
].iloc[0]

# Preserve actual model parameters
def clean_value(value):
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]

    return value


parameters = {
    key: clean_value(value)
    for key, value in model.get_params().items()
}

features = list(model.feature_names_in_)

# Per-horizon performance
horizon_results = []

for _, row in horizon_metrics.iterrows():
    horizon_results.append({
        "hours_ahead": int(row["hours_ahead"]),
        "direct_xgboost_mae": float(row["Direct XGBoost"]),
        "recursive_xgboost_mae": float(row["Recursive XGBoost"]),
        "previous_week_baseline_mae": float(row["Previous Week Baseline"]),
    })

# Calculate baseline improvement
mae_improvement = (
    (float(baseline_test["MAE"]) - float(direct_test["MAE"]))
    / float(baseline_test["MAE"])
) * 100

metadata = {
    "model_name": "MoveIQ Direct 24-Hour Demand Forecaster",
    "algorithm": "XGBoost Regressor",
    "forecasting_method": "Direct multi-horizon",
    "forecast_horizon_hours": 24,
    "model_file": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),

    "features": features,
    "feature_count": len(features),

    "training": {
        "training_samples": int(validation["training_samples"]),
        "validation_samples": int(validation["validation_samples"]),
        "training_seconds": float(validation["training_seconds"]),
        "parameters": parameters,
    },

    "validation_performance": {
        "mae": float(validation["validation_mae"]),
        "rmse": float(validation["validation_rmse"]),
        "mean_error": float(validation["validation_mean_error"]),
    },

    "untouched_test_performance": {
        "samples": int(direct_test["samples"]),
        "forecast_origins": 108,
        "horizons_per_origin": 24,
        "mae": float(direct_test["MAE"]),
        "rmse": float(direct_test["RMSE"]),
        "previous_week_baseline_mae": float(baseline_test["MAE"]),
        "previous_week_baseline_rmse": float(baseline_test["RMSE"]),
        "recursive_xgboost_mae": float(recursive_test["MAE"]),
        "recursive_xgboost_rmse": float(recursive_test["RMSE"]),
        "mae_improvement_over_previous_week_percent": round(mae_improvement, 2),
        "evaluation_type": "Rolling-origin 24-hour forecasts on held-out test period",
    },

    "horizon_performance": horizon_results,

    "model_selection": {
        "selected_model": "Direct XGBoost",
        "reason": (
            "Achieved lower overall MAE and RMSE than Recursive XGBoost "
            "and the previous-week baseline on the same untouched test forecasts."
        ),
    },

    "important_notes": [
        "The test set was not used for model training.",
        "Each forecast horizon is predicted independently.",
        "Predictions from earlier forecast hours are not fed into later predictions.",
        "Historical target_lag_24 and target_lag_168 features use information available at forecast origin.",
        "Performance metrics describe this dataset and evaluation period; they do not guarantee future accuracy.",
    ],
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, allow_nan=False)

print("=" * 65)
print("DIRECT MODEL FINALIZATION COMPLETE")
print("=" * 65)
print(f"Model: {metadata['model_name']}")
print(f"Algorithm: {metadata['algorithm']}")
print(f"Features: {len(features)}")
print(f"Training samples: {validation['training_samples']:,}")
print(f"Validation MAE: {validation['validation_mae']:.2f}")
print(f"Test MAE: {direct_test['MAE']:.2f}")
print(f"Test RMSE: {direct_test['RMSE']:.2f}")
print(f"Improvement over baseline: {mae_improvement:.2f}%")
print(f"\nMetadata saved to: {OUTPUT_PATH}")
