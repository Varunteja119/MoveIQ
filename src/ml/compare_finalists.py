from pathlib import Path
import joblib
import pandas as pd
import numpy as np

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

validation = pd.read_parquet(DATA_DIR / "validation.parquet")

models = {
    "XGBoost": "xgboost_tuned.joblib",
    "LightGBM": "lightgbm_tuned.joblib",
    "HistGradientBoosting": "histgradientboosting_tuned.joblib"
}

overall_results = []

for name, filename in models.items():

    print("\n" + "=" * 65)
    print(f"DIAGNOSTICS: {name}")
    print("=" * 65)

    model = joblib.load(MODEL_DIR / filename)

    result = validation.copy()
    result["prediction"] = model.predict(result[FEATURES])
    result["error"] = result["trip_count"] - result["prediction"]
    result["absolute_error"] = result["error"].abs()

    mae = mean_absolute_error(
        result["trip_count"], result["prediction"]
    )

    rmse = np.sqrt(mean_squared_error(
        result["trip_count"], result["prediction"]
    ))

    bias = result["error"].mean()

    overall_results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Mean error": round(bias, 2)
    })

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Mean error: {bias:.2f}")

    hourly = result.groupby("hour").agg(
        actual_average=("trip_count", "mean"),
        predicted_average=("prediction", "mean"),
        mae=("absolute_error", "mean")
    ).reset_index()

    weekday = result.groupby("day_of_week").agg(
        actual_average=("trip_count", "mean"),
        predicted_average=("prediction", "mean"),
        mae=("absolute_error", "mean")
    ).reset_index()

    worst = result.nlargest(10, "absolute_error")[
        ["pickup_hour", "trip_count", "prediction", "error", "absolute_error"]
    ]

    prefix = name.lower().replace(" ", "_")

    hourly.to_csv(
        REPORT_DIR / f"{prefix}_hourly_diagnostics.csv",
        index=False
    )

    weekday.to_csv(
        REPORT_DIR / f"{prefix}_weekday_diagnostics.csv",
        index=False
    )

    worst.to_csv(
        REPORT_DIR / f"{prefix}_largest_errors.csv",
        index=False
    )

    print("\nPerformance by hour:")
    print(hourly.round(2).to_string(index=False))

    print("\nPerformance by weekday (0=Sunday):")
    print(weekday.round(2).to_string(index=False))

comparison = pd.DataFrame(overall_results)

comparison.to_csv(
    REPORT_DIR / "finalist_model_diagnostics.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINALIST COMPARISON")
print("=" * 65)
print(comparison.to_string(index=False))

print("\nDiagnostic reports saved in reports/ml/")
