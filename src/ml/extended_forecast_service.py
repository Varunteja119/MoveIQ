from pathlib import Path

import joblib
import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml" / "processed"
MODELS = ROOT / "models"

CITY_HISTORY = DATA / "hourly_demand_features_2024_2026.parquet"
ZONE_HISTORY = DATA / "zone_hourly_demand_features_2024_2026.parquet"

CITY_MODEL = MODELS / "moveiq_direct_demand_forecaster_2024_2026_candidate.joblib"
ZONE_MODEL = MODELS / "moveiq_zone_direct_demand_forecaster_2024_2026_candidate.joblib"

CITY_FEATURES = [
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

ZONE_CATEGORICAL = ["zone_id", "borough", "service_zone"]

ZONE_NUMERIC = [
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

ZONE_FEATURES = ZONE_CATEGORICAL + ZONE_NUMERIC

HISTORY_COLUMNS = [
    "pickup_hour",
    "zone_id",
    "borough",
    "service_zone",
    "trip_count",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
]


class ExtendedForecastService:
    """Inference service for MoveIQ's 2024–2026 direct forecasting models."""

    def __init__(self):
        self._check_files()

        print("Loading extended candidate models...")

        self.city_model = joblib.load(CITY_MODEL)
        self.zone_model = joblib.load(ZONE_MODEL)

        self.city_history = pd.read_parquet(CITY_HISTORY)
        self.city_history["pickup_hour"] = pd.to_datetime(
            self.city_history["pickup_hour"]
        )

        self._validate_models()

        print("Extended forecasting models loaded successfully.")

    @staticmethod
    def _check_files():
        required = [
            CITY_HISTORY,
            ZONE_HISTORY,
            CITY_MODEL,
            ZONE_MODEL,
        ]

        missing = [str(path) for path in required if not path.exists()]

        if missing:
            raise FileNotFoundError(
                "Missing required forecasting files:\n"
                + "\n".join(missing)
            )

    def _validate_models(self):
        city_names = list(self.city_model.feature_names_in_)
        zone_names = list(self.zone_model.feature_names_in_)

        if city_names != CITY_FEATURES:
            raise ValueError(
                "City model feature order does not match inference features."
            )

        if zone_names != ZONE_FEATURES:
            raise ValueError(
                "Zone model feature order does not match inference features."
            )

    @staticmethod
    def _timestamp(value):
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is not None:
            raise ValueError(
                "Use a timezone-naive New York local timestamp. "
                "The source data uses local timestamps."
            )

        if timestamp != timestamp.floor("h"):
            raise ValueError(
                "Forecast origin must be an exact hourly timestamp."
            )

        return timestamp

    def _latest_zone_origin(self):
        """Find the latest complete origin matching six-hour training cadence."""

        con = duckdb.connect()

        try:
            path = ZONE_HISTORY.resolve().as_posix()

            result = con.execute(f"""
                WITH active_zones AS (
                    SELECT zone_id
                    FROM read_parquet('{path}')
                    GROUP BY zone_id
                    HAVING SUM(trip_count) > 0
                ),
                eligible AS (
                    SELECT
                        d.pickup_hour,
                        COUNT(DISTINCT d.zone_id) AS zone_count,
                        COUNT(*) FILTER (
                            WHERE d.lag_24 IS NULL
                               OR d.lag_168 IS NULL
                               OR d.rolling_mean_24 IS NULL
                               OR d.rolling_mean_168 IS NULL
                        ) AS invalid_rows
                    FROM read_parquet('{path}') d
                    INNER JOIN active_zones a USING (zone_id)
                    GROUP BY d.pickup_hour
                )
                SELECT MAX(pickup_hour)
                FROM eligible
                WHERE zone_count = 263
                  AND invalid_rows = 0
                  AND DATE_DIFF(
                      'hour',
                      TIMESTAMP '2024-01-01 00:00:00',
                      pickup_hour
                  ) % 6 = 0
            """).fetchone()[0]

        finally:
            con.close()

        if result is None:
            raise RuntimeError("No complete six-hour zone forecast origin found.")

        return pd.Timestamp(result)

    def _city_origin(self, requested=None):
        if requested is not None:
            origin = self._timestamp(requested)
        else:
            origin = self.city_history["pickup_hour"].max()

        row = self.city_history.loc[
            self.city_history["pickup_hour"] == origin
        ]

        if len(row) != 1:
            raise ValueError(f"Citywide origin not found: {origin}")

        row = row.iloc[0]

        required = [
            "trip_count",
            "lag_24",
            "lag_168",
            "rolling_mean_24",
            "rolling_mean_168",
        ]

        if row[required].isna().any():
            raise ValueError(
                f"Citywide historical features are incomplete at {origin}."
            )

        return origin, row

    def forecast_citywide(self, as_of=None):
        """Return the next 24 hourly citywide demand predictions."""

        origin, history = self._city_origin(as_of)
        records = []

        history_by_time = self.city_history.set_index("pickup_hour")

        for horizon in range(1, 25):
            target = origin + pd.Timedelta(hours=horizon)

            lag24_time = target - pd.Timedelta(hours=24)
            lag168_time = target - pd.Timedelta(hours=168)

            if lag24_time not in history_by_time.index:
                raise ValueError(f"Missing citywide lag-24 data for {target}")

            if lag168_time not in history_by_time.index:
                raise ValueError(f"Missing citywide lag-168 data for {target}")

            lag24 = history_by_time.loc[lag24_time, "trip_count"]
            lag168 = history_by_time.loc[lag168_time, "trip_count"]

            if isinstance(lag24, pd.Series) or isinstance(lag168, pd.Series):
                raise ValueError("Duplicate citywide historical timestamps found.")

            records.append({
                "target_hour": target.hour,
                "target_day_of_week": (target.day_of_week + 1) % 7,
                "target_is_weekend": int(target.day_of_week >= 5),
                "target_month": target.month,
                "hours_ahead": horizon,
                "origin_demand": history["trip_count"],
                "origin_lag_24": history["lag_24"],
                "origin_lag_168": history["lag_168"],
                "rolling_mean_24": history["rolling_mean_24"],
                "rolling_mean_168": history["rolling_mean_168"],
                "target_lag_24": lag24,
                "target_lag_168": lag168,
            })

        features = pd.DataFrame(records)[CITY_FEATURES]
        predictions = np.maximum(0, self.city_model.predict(features))

        result = pd.DataFrame({
            "forecast_origin": origin,
            "target_timestamp": [
                origin + pd.Timedelta(hours=h)
                for h in range(1, 25)
            ],
            "hours_ahead": range(1, 25),
            "predicted_demand": predictions,
        })

        result.attrs["model"] = "citywide_extended_candidate"
        return result

    def _load_zone_window(self, origin, zone_ids=None):
        """Load only the historical window and zones required for inference."""

        start = origin - pd.Timedelta(hours=168)
        end = origin

        filters = [
            ("pickup_hour", ">=", start),
            ("pickup_hour", "<=", end),
        ]

        if zone_ids is not None:
            filters.append(("zone_id", "in", zone_ids))

        history = pd.read_parquet(
            ZONE_HISTORY,
            columns=HISTORY_COLUMNS,
            filters=filters,
        )

        history["pickup_hour"] = pd.to_datetime(history["pickup_hour"])

        return history

    def _prepare_zone_features(self, origin, zone_history, category_snapshot):
        origin_rows = zone_history.loc[
            zone_history["pickup_hour"] == origin
        ].copy()

        if origin_rows.empty:
            raise ValueError(f"No zone history found at {origin}")

        if origin_rows["zone_id"].duplicated().any():
            raise ValueError("Duplicate zone records found at forecast origin.")

        required_origin = [
            "trip_count",
            "lag_24",
            "lag_168",
            "rolling_mean_24",
            "rolling_mean_168",
        ]

        if origin_rows[required_origin].isna().any().any():
            raise ValueError("Some zone origin features are incomplete.")

        # The model was trained with all active zones and these exact
        # categorical columns. Build category sets from the complete
        # origin snapshot before applying any requested-zone filtering.
        categories = {
            column: pd.Index(
                category_snapshot[column].dropna().unique()
            ).sort_values()
            for column in ZONE_CATEGORICAL
        }

        lookup = zone_history.set_index(
            ["zone_id", "pickup_hour"]
        )

        records = []

        for row in origin_rows.itertuples(index=False):
            zone_id = row.zone_id

            for horizon in range(1, 25):
                target = origin + pd.Timedelta(hours=horizon)

                lag24_time = target - pd.Timedelta(hours=24)
                lag168_time = target - pd.Timedelta(hours=168)

                key24 = (zone_id, lag24_time)
                key168 = (zone_id, lag168_time)

                if key24 not in lookup.index:
                    raise ValueError(
                        f"Missing lag-24 history for zone {zone_id}, "
                        f"target {target}"
                    )

                if key168 not in lookup.index:
                    raise ValueError(
                        f"Missing lag-168 history for zone {zone_id}, "
                        f"target {target}"
                    )

                lag24 = lookup.loc[key24, "trip_count"]
                lag168 = lookup.loc[key168, "trip_count"]

                if isinstance(lag24, pd.Series) or isinstance(lag168, pd.Series):
                    raise ValueError(
                        f"Duplicate historical records for zone {zone_id}."
                    )

                records.append({
                    "zone_id": zone_id,
                    "borough": row.borough,
                    "service_zone": row.service_zone,
                    "target_hour": target.hour,
                    "target_day_of_week": (target.day_of_week - 1) % 7,
                    "target_is_weekend": int(target.day_of_week >= 5),
                    "target_month": target.month,
                    "hours_ahead": horizon,
                    "origin_demand": row.trip_count,
                    "origin_lag_24": row.lag_24,
                    "origin_lag_168": row.lag_168,
                    "rolling_mean_24": row.rolling_mean_24,
                    "rolling_mean_168": row.rolling_mean_168,
                    "target_lag_24": lag24,
                    "target_lag_168": lag168,
                })

        features = pd.DataFrame(records)[ZONE_FEATURES]

        for column in ZONE_CATEGORICAL:
            features[column] = pd.Categorical(
                features[column],
                categories=categories[column],
            )

        return features, categories

    def forecast_zones(
        self,
        zone_ids=None,
        borough=None,
        as_of=None,
    ):
        """
        Forecast selected zones, all active zones, or one borough.

        Supply either zone_ids or borough, not both.
        """

        if zone_ids is not None and borough is not None:
            raise ValueError("Specify zone_ids or borough, not both.")

        origin = (
            self._timestamp(as_of)
            if as_of is not None
            else self._latest_zone_origin()
        )

        hours_from_anchor = int(
            (origin - pd.Timestamp("2024-01-01 00:00:00"))
            / pd.Timedelta(hours=1)
        )

        if hours_from_anchor % 6 != 0:
            raise ValueError(
                "Zone inference origin must follow the model's "
                "six-hour training cadence."
            )

        # Read the complete origin snapshot to establish the full
        # categorical domains, then load the requested historical window.
        con = duckdb.connect()
        try:
            zone_path = ZONE_HISTORY.resolve().as_posix()
            origin_snapshot = con.execute(f"""
                WITH active_zones AS (
                    SELECT zone_id
                    FROM read_parquet('{zone_path}')
                    GROUP BY zone_id
                    HAVING SUM(trip_count) > 0
                )
                SELECT
                    d.pickup_hour,
                    d.zone_id,
                    d.borough,
                    d.service_zone
                FROM read_parquet('{zone_path}') d
                INNER JOIN active_zones a USING (zone_id)
                WHERE d.pickup_hour = ?
            """, [origin.to_pydatetime()]).fetchdf()
        finally:
            con.close()

        if origin_snapshot["zone_id"].nunique() != 263:
            raise ValueError(
                f"Expected 263 active zones at {origin}; "
                f"found {origin_snapshot['zone_id'].nunique()}."
            )

        if zone_ids is not None:
            if isinstance(zone_ids, (int, np.integer)):
                zone_ids = [int(zone_ids)]
            else:
                zone_ids = [int(zone_id) for zone_id in zone_ids]

            if not zone_ids:
                raise ValueError("zone_ids cannot be empty.")

            available = set(origin_snapshot["zone_id"].astype(int))
            unknown = sorted(set(zone_ids) - available)

            if unknown:
                raise ValueError(f"Unknown or inactive zone IDs: {unknown}")

        elif borough is not None:
            matches = origin_snapshot.loc[
                origin_snapshot["borough"].str.casefold()
                == str(borough).casefold()
            ]

            if matches.empty:
                available = sorted(origin_snapshot["borough"].unique().tolist())
                raise ValueError(
                    f"Unknown borough '{borough}'. Available: {available}"
                )

            zone_ids = sorted(matches["zone_id"].astype(int).unique().tolist())

        else:
            zone_ids = sorted(
                origin_snapshot["zone_id"].astype(int).unique().tolist()
            )

        zone_history = self._load_zone_window(origin, zone_ids)

        # Use the full origin snapshot for category domains, while
        # predictions are restricted to the requested zones.
        selected_origin = origin_snapshot[
            origin_snapshot["zone_id"].isin(zone_ids)
        ]

        if selected_origin.empty:
            raise ValueError("No matching zones found.")

        features, _ = self._prepare_zone_features(origin, zone_history, origin_snapshot)

        features = features.loc[
            features["zone_id"].isin(zone_ids)
        ].copy()

        predictions = np.maximum(
            0,
            self.zone_model.predict(features[ZONE_FEATURES]),
        )

        result = features[
            ["zone_id", "borough", "service_zone", "hours_ahead"]
        ].copy()

        result["forecast_origin"] = origin
        result["target_timestamp"] = [
            origin + pd.Timedelta(hours=int(h))
            for h in result["hours_ahead"]
        ]
        result["predicted_demand"] = predictions

        result = result[
            [
                "forecast_origin",
                "target_timestamp",
                "hours_ahead",
                "zone_id",
                "borough",
                "service_zone",
                "predicted_demand",
            ]
        ].sort_values(["zone_id", "hours_ahead"]).reset_index(drop=True)

        result.attrs["model"] = "global_zone_extended_candidate"
        return result

    def forecast_borough(self, borough, as_of=None):
        """Forecast a borough by summing its zone-level predictions."""

        zone_forecasts = self.forecast_zones(
            borough=borough,
            as_of=as_of,
        )

        result = (
            zone_forecasts
            .groupby(
                ["forecast_origin", "target_timestamp", "hours_ahead", "borough"],
                as_index=False,
            )["predicted_demand"]
            .sum()
        )

        result = result.rename(
            columns={"predicted_demand": "predicted_demand"}
        )

        result.attrs["model"] = "borough_aggregate_of_global_zone_model"
        return result


if __name__ == "__main__":
    service = ExtendedForecastService()

    print("\nCITYWIDE FORECAST")
    print(service.forecast_citywide().to_string(index=False))

    print("\nSAMPLE ZONE FORECAST")
    print(
        service.forecast_zones(zone_ids=[1]).head(10).to_string(index=False)
    )
