from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=20,
        l2_regularization=1.0,
        random_state=42
    )
}

results = []

for name, model in models.items():
    print("\n" + "=" * 65)
    print(f"TRAINING: {name}")
    print("=" * 65)

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Training time (seconds)": round(training_time, 2)
    })

    model_filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / model_filename)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Saved: models/{model_filename}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "ml_model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("MODEL COMPARISON — VALIDATION DATA")
print("=" * 65)
print(results_df.to_string(index=False))

print("\nBASELINE MAE: 732.72")
print("BASELINE RMSE: 1119.77")

best_model = results_df.sort_values("MAE").iloc[0]

print(f"\nBest ML model by MAE: {best_model['Model']}")
print("\nResults saved to reports/ml/ml_model_comparison.csv")
