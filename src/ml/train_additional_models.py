from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

models = {
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        ))
    ])
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

    filename = name.lower() + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Saved: models/{filename}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "additional_model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("ADDITIONAL MODEL COMPARISON")
print("=" * 65)
print(results_df.sort_values("MAE").to_string(index=False))

print("\nEXISTING MODEL BENCHMARKS")
print("HistGradientBoosting: MAE 288.55 | RMSE 435.83")
print("Random Forest:        MAE 320.53 | RMSE 491.90")
print("Previous week:        MAE 732.72 | RMSE 1119.77")

print("\nResults saved to reports/ml/additional_model_comparison.csv")
