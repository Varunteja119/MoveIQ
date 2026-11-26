from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/ml/processed")
OUTPUT_DIR = Path("reports/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

validation = pd.read_parquet(DATA_DIR / "validation.parquet")

actual = validation["trip_count"]

predictions = {
    "Previous hour": validation["lag_1"],
    "Previous week": validation["lag_168"]
}

results = []

for name, prediction in predictions.items():
    valid = actual.notna() & prediction.notna()

    y_true = actual[valid]
    y_pred = prediction[valid]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results.append({
        "Model": name,
        "Records evaluated": len(y_true),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

results_df = pd.DataFrame(results)

print("=" * 65)
print("BASELINE MODEL EVALUATION — VALIDATION DATA")
print("=" * 65)
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "baseline_results.csv",
    index=False
)

print("\nResults saved to reports/ml/baseline_results.csv")
