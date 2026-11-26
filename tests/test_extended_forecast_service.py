import unittest

import numpy as np
import pandas as pd

from src.ml.extended_forecast_service import ExtendedForecastService


class TestExtendedForecastService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = ExtendedForecastService()
        cls.origin = pd.Timestamp("2026-09-30 18:00:00")

    def test_citywide_forecast_returns_24_hours(self):
        result = self.service.forecast_citywide(as_of=self.origin)

        self.assertEqual(len(result), 24)
        self.assertEqual(result["hours_ahead"].tolist(), list(range(1, 25)))
        self.assertTrue((result["predicted_demand"] >= 0).all())
        self.assertTrue((result["target_timestamp"] > self.origin).all())
        self.assertEqual(result["forecast_origin"].nunique(), 1)

    def test_selected_zone_returns_only_requested_zone(self):
        result = self.service.forecast_zones(
            zone_ids=[1],
            as_of=self.origin,
        )

        self.assertEqual(len(result), 24)
        self.assertEqual(result["zone_id"].nunique(), 1)
        self.assertEqual(int(result["zone_id"].iloc[0]), 1)

    def test_multiple_zone_selection(self):
        requested = {1, 132, 265}

        result = self.service.forecast_zones(
            zone_ids=list(requested),
            as_of=self.origin,
        )

        self.assertEqual(len(result), 72)
        self.assertEqual(set(result["zone_id"].unique()), requested)
        self.assertTrue(result.groupby("zone_id").size().eq(24).all())

    def test_all_active_zones(self):
        result = self.service.forecast_zones(as_of=self.origin)

        self.assertEqual(result["zone_id"].nunique(), 263)
        self.assertEqual(len(result), 263 * 24)
        self.assertTrue(result.groupby("zone_id").size().eq(24).all())
        self.assertTrue((result["predicted_demand"] >= 0).all())

    def test_borough_forecast(self):
        result = self.service.forecast_borough(
            "Manhattan",
            as_of=self.origin,
        )

        self.assertEqual(len(result), 24)
        self.assertTrue(
            result["borough"].str.casefold().eq("manhattan").all()
        )
        self.assertTrue((result["predicted_demand"] >= 0).all())

    def test_borough_aggregation_matches_zone_sum(self):
        zones = self.service.forecast_zones(
            borough="Manhattan",
            as_of=self.origin,
        )

        borough = self.service.forecast_borough(
            "Manhattan",
            as_of=self.origin,
        )

        expected = (
            zones.groupby("hours_ahead")["predicted_demand"]
            .sum()
            .sort_index()
            .to_numpy()
        )

        actual = (
            borough.set_index("hours_ahead")["predicted_demand"]
            .sort_index()
            .to_numpy()
        )

        np.testing.assert_allclose(
            expected,
            actual,
            rtol=1e-5,
            atol=1e-4,
        )

    def test_invalid_zone_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.forecast_zones(
                zone_ids=[99999],
                as_of=self.origin,
            )

    def test_invalid_borough_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.forecast_borough(
                "Atlantis",
                as_of=self.origin,
            )

    def test_non_aligned_zone_origin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.forecast_zones(
                zone_ids=[1],
                as_of="2026-09-30 19:00:00",
            )

    def test_missing_citywide_origin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.forecast_citywide(
                as_of="2026-10-01 00:00:00",
            )


    def test_non_hour_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.forecast_citywide(
                as_of="2026-09-30 18:45:00"
            )

    def test_timezone_aware_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.forecast_zones(
                zone_ids=[1],
                as_of="2026-09-30T18:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)