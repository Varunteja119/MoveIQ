from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "ml" / "processed"
    / "hourly_demand_2024_2025.parquet"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "moveiq_demand_forecaster.joblib"
)

FEATURES = [
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


class MoveIQForecastService:
    def __init__(
        self,
        history_path=DEFAULT_HISTORY_PATH,
        model_path=DEFAULT_MODEL_PATH,
    ):
        self.history_path = Path(history_path)
        self.model_path = Path(model_path)

        if not self.history_path.exists():
            raise FileNotFoundError(
                f"Historical data not found: {self.history_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Forecasting model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)
        self.history = self._load_history()

    def _load_history(self):
        history = pd.read_parquet(self.history_path)

        required_columns = {"pickup_hour", "trip_count"}
        missing = required_columns - set(history.columns)

        if missing:
            raise ValueError(
                f"Historical data is missing columns: {sorted(missing)}"
            )

        history = history[["pickup_hour", "trip_count"]].copy()
        history["pickup_hour"] = pd.to_datetime(history["pickup_hour"])
        history["trip_count"] = pd.to_numeric(
            history["trip_count"], errors="raise"
        )

        history = history.sort_values("pickup_hour").reset_index(drop=True)

        if history.empty:
            raise ValueError("Historical dataset is empty.")

        if history["pickup_hour"].duplicated().any():
            raise ValueError("Duplicate hourly timestamps found.")

        if history["trip_count"].isna().any():
            raise ValueError("Missing trip_count values found.")

        return history

    def _build_features(self, history, target_timestamp):
        history = history.sort_values("pickup_hour").reset_index(drop=True)
        target_timestamp = pd.Timestamp(target_timestamp)

        # Matches fix_lag_feature.py:
        # previous available observation, not necessarily timestamp - 1 hour.
        previous_row = history.iloc[-1]

        # DuckDB EXTRACT(DOW): Sunday=0, Monday=1, ..., Saturday=6.
        day_of_week = (target_timestamp.day_of_week + 1) % 7

        timestamp_lookup = dict(
            zip(history["pickup_hour"], history["trip_count"])
        )

        lag_24_timestamp = target_timestamp - pd.Timedelta(hours=24)
        lag_168_timestamp = target_timestamp - pd.Timedelta(hours=168)

        if lag_24_timestamp not in timestamp_lookup:
            raise ValueError(
                f"Cannot build lag_24 for {target_timestamp}: "
                f"{lag_24_timestamp} is missing from history."
            )

        if lag_168_timestamp not in timestamp_lookup:
            raise ValueError(
                f"Cannot build lag_168 for {target_timestamp}: "
                f"{lag_168_timestamp} is missing from history."
            )

        # Match build_features.py exactly:
        # previous.pickup_hour < target_timestamp
        # previous.pickup_hour >= target_timestamp - interval
        rolling_24_start = target_timestamp - pd.Timedelta(hours=24)
        rolling_168_start = target_timestamp - pd.Timedelta(hours=168)

        rolling_24 = history.loc[
            (history["pickup_hour"] < target_timestamp)
            & (history["pickup_hour"] >= rolling_24_start),
            "trip_count",
        ]

        rolling_168 = history.loc[
            (history["pickup_hour"] < target_timestamp)
            & (history["pickup_hour"] >= rolling_168_start),
            "trip_count",
        ]

        if rolling_24.empty or rolling_168.empty:
            raise ValueError(
                f"Insufficient rolling-window history for {target_timestamp}."
            )

        features = {
            "hour": target_timestamp.hour,
            "day_of_week": day_of_week,
            "is_weekend": int(day_of_week in [0, 6]),
            "month": target_timestamp.month,
            "lag_1": float(previous_row["trip_count"]),
            "lag_24": float(timestamp_lookup[lag_24_timestamp]),
            "lag_168": float(timestamp_lookup[lag_168_timestamp]),
            "rolling_mean_24": float(rolling_24.mean()),
            "rolling_mean_168": float(rolling_168.mean()),
        }

        return pd.DataFrame([features], columns=FEATURES)

    def forecast_next_24_hours(self):
        forecast_history = self.history.copy()

        last_observed = forecast_history.iloc[-1]
        last_timestamp = pd.Timestamp(last_observed["pickup_hour"])

        forecasts = []

        for step in range(1, 25):
            target_timestamp = last_timestamp + pd.Timedelta(hours=step)

            feature_row = self._build_features(
                forecast_history,
                target_timestamp,
            )

            prediction = float(self.model.predict(feature_row)[0])
            prediction = max(0.0, prediction)

            forecasts.append({
                "forecast_timestamp": target_timestamp,
                "hours_ahead": step,
                "predicted_pickups": round(prediction, 2),
            })

            forecast_history = pd.concat(
                [
                    forecast_history,
                    pd.DataFrame([{
                        "pickup_hour": target_timestamp,
                        "trip_count": prediction,
                    }]),
                ],
                ignore_index=True,
            )

        result = pd.DataFrame(forecasts)

        result.attrs["last_observed_timestamp"] = last_timestamp
        result.attrs["last_observed_pickups"] = float(
            last_observed["trip_count"]
        )
        result.attrs["model"] = "MoveIQ Hourly Demand Forecaster"
        result.attrs["forecasting_method"] = "Recursive 24-hour forecast"

        return result


if __name__ == "__main__":
    service = MoveIQForecastService()
    forecast = service.forecast_next_24_hours()

    print("\nMOVEIQ - NEXT 24 HOURS DEMAND FORECAST")
    print("=" * 65)
    print(f"Last observed hour : {forecast.attrs['last_observed_timestamp']}")
    print(f"Observed pickups   : {forecast.attrs['last_observed_pickups']:,.0f}")
    print(f"Forecast method    : {forecast.attrs['forecasting_method']}")
    print("=" * 65)

    print(forecast.to_string(index=False))

    peak = forecast.loc[forecast["predicted_pickups"].idxmax()]

    print("\nFORECAST SUMMARY")
    print("-" * 65)
    print(
        f"Expected total pickups : "
        f"{forecast['predicted_pickups'].sum():,.0f}"
    )
    print(
        f"Expected average/hour  : "
        f"{forecast['predicted_pickups'].mean():,.0f}"
    )
    print(
        f"Predicted peak hour    : "
        f"{peak['forecast_timestamp']} "
        f"({peak['predicted_pickups']:,.0f} pickups)"
    )

    output_path = (
        PROJECT_ROOT / "reports" / "ml" / "next_24_hour_forecast.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)

    print(f"\nForecast saved to: {output_path}")
