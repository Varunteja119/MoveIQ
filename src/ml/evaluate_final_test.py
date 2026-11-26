from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/ml/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/ml")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

test = pd.read_parquet(DATA_DIR / "test.parquet")

# Load the models selected before final testing.
models = {
    "Original XGBoost": joblib.load(MODEL_DIR / "xgboost.joblib"),
    "Tuned XGBoost": joblib.load(MODEL_DIR / "xgboost_tuned.joblib")
}

actual = test["trip_count"]

predictions = {
    "Previous hour": test["lag_1"].to_numpy(),
    "Previous week": test["lag_168"].to_numpy()
}

for name, model in models.items():
    predictions[name] = model.predict(test[FEATURES])

results = []

prediction_data = test[
    ["pickup_hour", "trip_count", "hour", "day_of_week"]
].copy()

for name, prediction in predictions.items():

    valid = np.isfinite(prediction) & actual.notna().to_numpy()

    y_true = actual.to_numpy()[valid]
    y_pred = prediction[valid]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_true - y_pred)

    results.append({
        "Model": name,
        "Records evaluated": len(y_true),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Mean error": round(bias, 2)
    })

    prediction_data[name] = prediction

results_df = pd.DataFrame(results).sort_values("MAE")

print("=" * 70)
print("FINAL TEST RESULTS — UNSEEN DATA")
print("=" * 70)

print(f"Test period: {test['pickup_hour'].min()} to {test['pickup_hour'].max()}")
print(f"Test records: {len(test):,}")

print("\n" + results_df.to_string(index=False))

# Performance of the selected model by hour.
prediction_data["absolute_error"] = (
    prediction_data["trip_count"] -
    prediction_data["Tuned XGBoost"]
).abs()

hourly = prediction_data.groupby("hour").agg(
    actual_average=("trip_count", "mean"),
    predicted_average=("Tuned XGBoost", "mean"),
    mae=("absolute_error", "mean")
).reset_index()

print("\nTUNED XGBOOST — TEST PERFORMANCE BY HOUR")
print(hourly.round(2).to_string(index=False))

# Save test results and predictions.
results_df.to_csv(
    REPORT_DIR / "final_test_results.csv",
    index=False
)

hourly.to_csv(
    REPORT_DIR / "final_test_hourly_performance.csv",
    index=False
)

prediction_data.to_csv(
    REPORT_DIR / "final_test_predictions.csv",
    index=False
)

print("\nSaved:")
print("- reports/ml/final_test_results.csv")
print("- reports/ml/final_test_hourly_performance.csv")
print("- reports/ml/final_test_predictions.csv")

print("\nFinal test evaluation completed.")
