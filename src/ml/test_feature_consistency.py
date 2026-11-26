from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.forecast_service import MoveIQForecastService, FEATURES


ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT / "data" / "ml" / "processed"
    / "hourly_demand_features.parquet"
)


def main():
    service = MoveIQForecastService()

    saved_features = pd.read_parquet(FEATURE_PATH)
    saved_features["pickup_hour"] = pd.to_datetime(
        saved_features["pickup_hour"]
    )

    # Include ordinary dates and both spring DST transitions.
    test_timestamps = [
        pd.Timestamp("2024-06-15 12:00:00"),
        pd.Timestamp("2024-03-10 03:00:00"),
        pd.Timestamp("2025-03-09 03:00:00"),
        pd.Timestamp("2025-06-15 18:00:00"),
        pd.Timestamp("2025-12-15 08:00:00"),
        pd.Timestamp("2025-12-31 23:00:00"),
    ]

    print("\nMOVEIQ FEATURE CONSISTENCY TEST")
    print("=" * 65)

    for timestamp in test_timestamps:
        expected_rows = saved_features.loc[
            saved_features["pickup_hour"] == timestamp
        ]

        if expected_rows.empty:
            raise AssertionError(
                f"Timestamp missing from saved features: {timestamp}"
            )

        expected = expected_rows.iloc[0]

        if any(pd.isna(expected[column]) for column in FEATURES):
            raise AssertionError(
                f"Saved training features contain nulls at {timestamp}"
            )

        historical = service.history.loc[
            service.history["pickup_hour"] < timestamp
        ].copy()

        actual = service._build_features(
            historical,
            timestamp,
        ).iloc[0]

        for column in FEATURES:
            expected_value = float(expected[column])
            actual_value = float(actual[column])

            if not np.isclose(
                expected_value,
                actual_value,
                rtol=1e-9,
                atol=1e-6,
            ):
                raise AssertionError(
                    f"Mismatch at {timestamp}, feature={column}: "
                    f"training={expected_value}, service={actual_value}"
                )

        print(f"PASS: {timestamp}")

    print("\nAll selected features match the training dataset.")


if __name__ == "__main__":
    main()
