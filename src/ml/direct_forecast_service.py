from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "hourly_demand_2024_2025.parquet"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "moveiq_direct_demand_forecaster.joblib"
)

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


class DirectForecastService:
    """
    Generates independent 24-hour demand forecasts using
    the MoveIQ Direct XGBoost model.
    """

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
                f"Direct forecasting model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)
        self.history = self._load_history()

        model_features = list(self.model.feature_names_in_)

        if model_features != FEATURES:
            raise ValueError(
                "Model feature order does not match the service feature order."
            )

    def _load_history(self):
        history = pd.read_parquet(self.history_path)

        required_columns = {"pickup_hour", "trip_count"}
        missing = required_columns - set(history.columns)

        if missing:
            raise ValueError(
                f"Historical data is missing columns: {sorted(missing)}"
            )

        history = history[["pickup_hour", "trip_count"]].copy()

        history["pickup_hour"] = pd.to_datetime(
            history["pickup_hour"]
        )

        history["trip_count"] = pd.to_numeric(
            history["trip_count"],
            errors="raise",
        )

        history = (
            history
            .sort_values("pickup_hour")
            .reset_index(drop=True)
        )

        if history.empty:
            raise ValueError("Historical demand dataset is empty.")

        if history["pickup_hour"].duplicated().any():
            raise ValueError("Duplicate hourly timestamps found.")

        if history["trip_count"].isna().any():
            raise ValueError("Missing trip_count values found.")

        if (history["trip_count"] < 0).any():
            raise ValueError("Negative historical demand values found.")

        return history

    def _build_origin_features(self, history, origin_timestamp):
        """
        Calculate features available at the forecast origin.

        Rolling windows use timestamps:
        origin - interval <= timestamp < origin.
        """

        origin_timestamp = pd.Timestamp(origin_timestamp)

        timestamp_lookup = dict(
            zip(history["pickup_hour"], history["trip_count"])
        )

        required_lag_24 = (
            origin_timestamp - pd.Timedelta(hours=24)
        )

        required_lag_168 = (
            origin_timestamp - pd.Timedelta(hours=168)
        )

        if required_lag_24 not in timestamp_lookup:
            raise ValueError(
                f"Missing origin lag_24 timestamp: {required_lag_24}"
            )

        if required_lag_168 not in timestamp_lookup:
            raise ValueError(
                f"Missing origin lag_168 timestamp: {required_lag_168}"
            )

        rolling_24_start = (
            origin_timestamp - pd.Timedelta(hours=24)
        )

        rolling_168_start = (
            origin_timestamp - pd.Timedelta(hours=168)
        )

        rolling_24 = history.loc[
            (history["pickup_hour"] < origin_timestamp)
            & (history["pickup_hour"] >= rolling_24_start),
            "trip_count",
        ]

        rolling_168 = history.loc[
            (history["pickup_hour"] < origin_timestamp)
            & (history["pickup_hour"] >= rolling_168_start),
            "trip_count",
        ]

        if rolling_24.empty or rolling_168.empty:
            raise ValueError(
                "Insufficient historical data for rolling features."
            )

        origin_row = history.loc[
            history["pickup_hour"] == origin_timestamp
        ].iloc[0]

        return {
            "origin_demand": float(origin_row["trip_count"]),
            "origin_lag_24": float(timestamp_lookup[required_lag_24]),
            "origin_lag_168": float(timestamp_lookup[required_lag_168]),
            "rolling_mean_24": float(rolling_24.mean()),
            "rolling_mean_168": float(rolling_168.mean()),
        }

    def _build_forecast_features(
        self,
        history,
        origin_timestamp,
        origin_features,
    ):
        """
        Construct all 24 feature rows using only information
        available at the forecast origin.
        """

        origin_timestamp = pd.Timestamp(origin_timestamp)

        timestamp_lookup = dict(
            zip(history["pickup_hour"], history["trip_count"])
        )

        feature_rows = []

        for hours_ahead in range(1, 25):

            target_timestamp = (
                origin_timestamp
                + pd.Timedelta(hours=hours_ahead)
            )

            target_lag_24_timestamp = (
                target_timestamp - pd.Timedelta(hours=24)
            )

            target_lag_168_timestamp = (
                target_timestamp - pd.Timedelta(hours=168)
            )

            # These historical timestamps must be available
            # at the forecast origin. No future actuals are used.
            if target_lag_24_timestamp > origin_timestamp:
                raise ValueError(
                    "Target lag_24 would require future information."
                )

            if target_lag_168_timestamp > origin_timestamp:
                raise ValueError(
                    "Target lag_168 would require future information."
                )

            if target_lag_24_timestamp not in timestamp_lookup:
                raise ValueError(
                    f"Missing historical target_lag_24: "
                    f"{target_lag_24_timestamp}"
                )

            if target_lag_168_timestamp not in timestamp_lookup:
                raise ValueError(
                    f"Missing historical target_lag_168: "
                    f"{target_lag_168_timestamp}"
                )

            # DuckDB EXTRACT(DOW):
            # Sunday=0, Monday=1, ..., Saturday=6.
            target_day_of_week = (
                target_timestamp.day_of_week + 1
            ) % 7

            feature_rows.append({
                "target_hour": target_timestamp.hour,
                "target_day_of_week": target_day_of_week,
                "target_is_weekend": int(
                    target_day_of_week in [0, 6]
                ),
                "target_month": target_timestamp.month,
                "hours_ahead": hours_ahead,

                "origin_demand": origin_features["origin_demand"],
                "origin_lag_24": origin_features["origin_lag_24"],
                "origin_lag_168": origin_features["origin_lag_168"],
                "rolling_mean_24": origin_features["rolling_mean_24"],
                "rolling_mean_168": origin_features["rolling_mean_168"],

                "target_lag_24": float(
                    timestamp_lookup[target_lag_24_timestamp]
                ),
                "target_lag_168": float(
                    timestamp_lookup[target_lag_168_timestamp]
                ),
            })

        return pd.DataFrame(
            feature_rows,
            columns=FEATURES,
        )

    def forecast_next_24_hours(self):
        """
        Generate 24 independent hourly predictions.

        Earlier predictions are never used as inputs
        for later forecast hours.
        """

        history = self.history.copy()

        last_observed = history.iloc[-1]
        origin_timestamp = pd.Timestamp(
            last_observed["pickup_hour"]
        )

        origin_features = self._build_origin_features(
            history,
            origin_timestamp,
        )

        feature_data = self._build_forecast_features(
            history,
            origin_timestamp,
            origin_features,
        )

        # One batch prediction for all 24 forecast horizons.
        predictions = self.model.predict(feature_data)

        predictions = [
            max(0.0, float(value))
            for value in predictions
        ]

        forecast_timestamps = [
            origin_timestamp + pd.Timedelta(hours=step)
            for step in range(1, 25)
        ]

        result = pd.DataFrame({
            "forecast_timestamp": forecast_timestamps,
            "hours_ahead": range(1, 25),
            "predicted_pickups": [
                round(value, 2) for value in predictions
            ],
        })

        result.attrs["last_observed_timestamp"] = origin_timestamp
        result.attrs["last_observed_pickups"] = float(
            last_observed["trip_count"]
        )
        result.attrs["model"] = (
            "MoveIQ Direct 24-Hour Demand Forecaster"
        )
        result.attrs["forecasting_method"] = (
            "Direct multi-horizon XGBoost"
        )
        result.attrs["forecast_horizon"] = 24

        return result


if __name__ == "__main__":

    service = DirectForecastService()

    forecast = service.forecast_next_24_hours()

    print("\nMOVEIQ - DIRECT 24-HOUR DEMAND FORECAST")
    print("=" * 75)

    print(
        f"Last observed hour : "
        f"{forecast.attrs['last_observed_timestamp']}"
    )

    print(
        f"Observed pickups   : "
        f"{forecast.attrs['last_observed_pickups']:,.0f}"
    )

    print(
        f"Forecast method    : "
        f"{forecast.attrs['forecasting_method']}"
    )

    print("=" * 75)

    print(forecast.to_string(index=False))

    peak = forecast.loc[
        forecast["predicted_pickups"].idxmax()
    ]

    print("\nFORECAST SUMMARY")
    print("-" * 75)

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
        PROJECT_ROOT
        / "reports"
        / "ml"
        / "direct_next_24_hour_forecast.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(output_path, index=False)

    print(f"\nForecast saved to: {output_path}")
