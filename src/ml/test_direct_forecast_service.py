from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.ml.direct_forecast_service import DirectForecastService, FEATURES


ROOT = Path(__file__).resolve().parents[2]

HISTORY_FILE = (
    ROOT / "data" / "ml" / "processed"
    / "hourly_demand_2024_2025.parquet"
)

DIRECT_DATASET_FILE = (
    ROOT / "data" / "ml" / "processed"
    / "direct_forecasting_samples.parquet"
)


class TestDirectForecastService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = DirectForecastService()

    def test_forecast_returns_exactly_24_predictions(self):
        forecast = self.service.forecast_next_24_hours()

        self.assertEqual(len(forecast), 24)
        self.assertEqual(
            forecast["hours_ahead"].tolist(),
            list(range(1, 25)),
        )

    def test_forecast_timestamps_are_ordered_and_unique(self):
        forecast = self.service.forecast_next_24_hours()

        timestamps = forecast["forecast_timestamp"]

        self.assertTrue(timestamps.is_monotonic_increasing)
        self.assertFalse(timestamps.duplicated().any())

        expected = pd.date_range(
            start=(
                forecast.attrs["last_observed_timestamp"]
                + pd.Timedelta(hours=1)
            ),
            periods=24,
            freq="h",
        )

        self.assertTrue(
            timestamps.reset_index(drop=True).equals(
                pd.Series(expected, name="forecast_timestamp")
            )
        )

    def test_predictions_are_finite_and_nonnegative(self):
        forecast = self.service.forecast_next_24_hours()

        predictions = forecast["predicted_pickups"].to_numpy()

        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())

    def test_feature_consistency_with_training_dataset(self):
        history = pd.read_parquet(HISTORY_FILE)
        training_samples = pd.read_parquet(DIRECT_DATASET_FILE)

        # Select a known origin from the original training dataset.
        sample_origin = training_samples[
            "origin_timestamp"
        ].iloc[0]

        sample_origin = pd.Timestamp(sample_origin)

        historical_data = history[
            history["pickup_hour"] <= sample_origin
        ].copy()

        origin_features = self.service._build_origin_features(
            historical_data,
            sample_origin,
        )

        generated_features = self.service._build_forecast_features(
            historical_data,
            sample_origin,
            origin_features,
        )

        expected_features = training_samples[
            training_samples["origin_timestamp"] == sample_origin
        ].sort_values("hours_ahead")[FEATURES].reset_index(drop=True)

        generated_features = generated_features.reset_index(drop=True)

        self.assertEqual(len(expected_features), 24)

        pd.testing.assert_frame_equal(
            generated_features,
            expected_features,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )

    def test_future_actual_values_do_not_change_features(self):
        history = self.service.history.copy()

        origin_timestamp = pd.Timestamp(
            history["pickup_hour"].iloc[-1]
        )

        origin_features = self.service._build_origin_features(
            history,
            origin_timestamp,
        )

        original_features = self.service._build_forecast_features(
            history,
            origin_timestamp,
            origin_features,
        )

        # Add artificial future records with deliberately extreme values.
        future_rows = pd.DataFrame({
            "pickup_hour": [
                origin_timestamp + pd.Timedelta(hours=i)
                for i in range(1, 25)
            ],
            "trip_count": [999999.0] * 24,
        })

        history_with_future_values = pd.concat(
            [history, future_rows],
            ignore_index=True,
        )

        modified_origin_features = (
            self.service._build_origin_features(
                history_with_future_values,
                origin_timestamp,
            )
        )

        modified_features = self.service._build_forecast_features(
            history_with_future_values,
            origin_timestamp,
            modified_origin_features,
        )

        pd.testing.assert_frame_equal(
            original_features,
            modified_features,
            check_dtype=False,
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
