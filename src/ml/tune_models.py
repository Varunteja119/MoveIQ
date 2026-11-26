from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

DATA_DIR = Path("data/ml/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/ml")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
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

TARGET = "trip_count"

train = pd.read_parquet(DATA_DIR / "train.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

time_split = TimeSeriesSplit(n_splits=3)

searches = {
    "XGBoost": (
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": [200, 400],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.05]
        }
    ),

    "LightGBM": (
        LGBMRegressor(
            random_state=42,
            n_jobs=1,
            verbosity=-1
        ),
        {
            "n_estimators": [200, 400],
            "num_leaves": [15, 31],
            "learning_rate": [0.03, 0.05]
        }
    ),

    "HistGradientBoosting": (
        HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=False
        ),
        {
            "max_iter": [150, 250],
            "learning_rate": [0.05, 0.08],
            "max_leaf_nodes": [15, 31]
        }
    )
}

results = []

for name, (model, parameters) in searches.items():

    print("\n" + "=" * 65)
    print(f"TUNING: {name}")
    print("=" * 65)

    start = time.time()

    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="neg_mean_absolute_error",
        cv=time_split,
        n_jobs=1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    tuning_time = time.time() - start

    best_model = search.best_estimator_
    predictions = best_model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    filename = name.lower().replace(" ", "_") + "_tuned.joblib"
    joblib.dump(best_model, MODEL_DIR / filename)

    results.append({
        "Model": name,
        "CV MAE": round(-search.best_score_, 2),
        "Validation MAE": round(mae, 2),
        "Validation RMSE": round(rmse, 2),
        "Tuning time (seconds)": round(tuning_time, 2),
        "Best parameters": str(search.best_params_)
    })

    print("\nBest parameters:", search.best_params_)
    print(f"Cross-validation MAE: {-search.best_score_:.2f}")
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    print(f"Saved: models/{filename}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Validation MAE")

results_df.to_csv(
    REPORT_DIR / "tuned_model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("TUNED MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nPREVIOUS BEST VALIDATION MAE")
print("XGBoost: 277.57")
print("LightGBM: 280.44")
print("HistGradientBoosting: 288.55")

print("\nResults saved to reports/ml/tuned_model_comparison.csv")
