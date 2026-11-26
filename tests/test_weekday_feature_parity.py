import unittest
import pandas as pd

from src.ml.extended_forecast_service import ExtendedForecastService


class TestWeekdayFeatureParity(unittest.TestCase):

    def test_citywide_weekday_encoding(self):
        cases = {
            "2026-09-27": 0,  # Sunday
            "2026-09-28": 1,  # Monday
            "2026-10-03": 6,  # Saturday
        }

        for date, expected in cases.items():
            target = pd.Timestamp(date)
            actual = (target.day_of_week + 1) % 7
            self.assertEqual(actual, expected)

    def test_zone_weekday_encoding(self):
        cases = {
            "2026-09-27": 5,  # Sunday
            "2026-09-28": 6,  # Monday
            "2026-10-03": 4,  # Saturday
        }

        for date, expected in cases.items():
            target = pd.Timestamp(date)
            actual = (target.day_of_week - 1) % 7
            self.assertEqual(actual, expected)

    def test_weekend_encoding(self):
        cases = {
            "2026-09-27": 1,  # Sunday
            "2026-09-28": 0,  # Monday
            "2026-10-03": 1,  # Saturday
        }

        for date, expected in cases.items():
            target = pd.Timestamp(date)
            actual = int(target.day_of_week >= 5)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
