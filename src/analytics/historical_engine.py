from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORICAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "analytics"
    / "historical"
    / "zone_hourly_2024_2026.parquet"
)


class HistoricalAnalytics:

    def __init__(self, dataset_path=DEFAULT_HISTORICAL_DATASET):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Historical dataset not found: {self.dataset_path}"
            )

        self.connection = duckdb.connect(database=":memory:")

        self.connection.execute("""
            SET memory_limit = '256MB'
        """)

        self.connection.execute("""
            SET threads = 1
        """)

        dataset_path = str(self.dataset_path).replace("'", "''")
        self.connection.execute(
            f"""
            CREATE VIEW historical_trips AS
            SELECT *
            FROM read_parquet('{dataset_path}')
            """
        )

        print("MoveIQ Historical Analytics initialized.")

    @staticmethod
    def _validate_filters(location_role, start_date, end_date):
        if location_role not in {"pickup", "dropoff"}:
            raise ValueError(
                "location_role must be 'pickup' or 'dropoff'."
            )

        if start_date and end_date and start_date > end_date:
            raise ValueError(
                "start_date must be earlier than or equal to end_date."
            )

    def _conditions(
        self,
        location_id=None,
        location_role="pickup",
        start_date=None,
        end_date=None,
        hour=None,
    ):
        self._validate_filters(
            location_role,
            start_date,
            end_date,
        )

        conditions = ["location_role = ?"]
        params = [location_role]

        if location_id is not None:
            conditions.append("location_id = ?")
            params.append(int(location_id))

        if start_date:
            conditions.append("pickup_date >= CAST(? AS DATE)")
            params.append(start_date)

        if end_date:
            conditions.append("pickup_date <= CAST(? AS DATE)")
            params.append(end_date)

        if hour is not None:
            if not 0 <= int(hour) <= 23:
                raise ValueError("hour must be between 0 and 23.")

            conditions.append("pickup_hour_of_day = ?")
            params.append(int(hour))

        return " AND ".join(conditions), params

    @staticmethod
    def _metrics_sql():
        return """
            SUM(total_trips) AS total_trips,

            SUM(valid_fare_trips) AS valid_fare_trips,
            SUM(fare_sum) AS fare_revenue,

            CASE
                WHEN SUM(valid_fare_trips) > 0
                THEN SUM(fare_sum) / SUM(valid_fare_trips)
                ELSE NULL
            END AS average_fare,

            SUM(valid_distance_trips) AS valid_distance_trips,

            CASE
                WHEN SUM(valid_distance_trips) > 0
                THEN SUM(distance_sum) / SUM(valid_distance_trips)
                ELSE NULL
            END AS average_distance_miles,

            SUM(valid_duration_trips) AS valid_duration_trips,

            CASE
                WHEN SUM(valid_duration_trips) > 0
                THEN SUM(duration_seconds_sum)
                     / SUM(valid_duration_trips) / 60.0
                ELSE NULL
            END AS average_duration_minutes
        """

    def get_overview(self):
        result = self.connection.execute("""
            SELECT
                MIN(pickup_date) AS first_date,
                MAX(pickup_date) AS last_date,
                COUNT(DISTINCT location_id) AS zone_count,
                SUM(total_trips) AS total_trips,
                COUNT(*) AS aggregate_rows
            FROM historical_trips
            WHERE location_role = 'pickup'
        """).fetchdf()

        return result.to_dict(orient="records")[0]

    def get_zone_profile(
        self,
        location_id,
        location_role="pickup",
        start_date=None,
        end_date=None,
    ):
        where, params = self._conditions(
            location_id=location_id,
            location_role=location_role,
            start_date=start_date,
            end_date=end_date,
        )

        query = f"""
            SELECT
                location_id,
                zone_name,
                borough,
                MIN(pickup_date) AS first_date,
                MAX(pickup_date) AS last_date,
                {self._metrics_sql()}
            FROM historical_trips
            WHERE {where}
            GROUP BY location_id, zone_name, borough
        """

        result = self.connection.execute(
            query,
            params,
        ).fetchdf()

        if result.empty:
            return None

        return result.to_dict(orient="records")[0]

    def get_zone_hourly(
        self,
        location_id,
        location_role="pickup",
        start_date=None,
        end_date=None,
    ):
        where, params = self._conditions(
            location_id=location_id,
            location_role=location_role,
            start_date=start_date,
            end_date=end_date,
        )

        query = f"""
            SELECT
                pickup_hour,
                pickup_date,
                pickup_hour_of_day,
                {self._metrics_sql()}
            FROM historical_trips
            WHERE {where}
            GROUP BY
                pickup_hour,
                pickup_date,
                pickup_hour_of_day
            ORDER BY pickup_hour
        """

        result = self.connection.execute(
            query,
            params,
        ).fetchdf()

        return result.to_dict(orient="records")

    def compare_periods(
        self,
        location_id,
        segment_a_start,
        segment_a_end,
        segment_b_start,
        segment_b_end,
        location_role="pickup",
        hour=None,
    ):
        where, base_params = self._conditions(
            location_id=location_id,
            location_role=location_role,
            hour=hour,
        )

        query = f"""
            SELECT
                CASE
                    WHEN pickup_date BETWEEN
                        CAST(? AS DATE) AND CAST(? AS DATE)
                    THEN 'segment_a'
                    ELSE 'segment_b'
                END AS segment,
                {self._metrics_sql()}
            FROM historical_trips
            WHERE {where}
              AND (
                    pickup_date BETWEEN
                        CAST(? AS DATE) AND CAST(? AS DATE)
                    OR
                    pickup_date BETWEEN
                        CAST(? AS DATE) AND CAST(? AS DATE)
              )
            GROUP BY segment
            ORDER BY segment
        """

        params = [
            segment_a_start,
            segment_a_end,
            *base_params,
            segment_a_start,
            segment_a_end,
            segment_b_start,
            segment_b_end,
        ]

        result = self.connection.execute(
            query,
            params,
        ).fetchdf()

        return result.to_dict(orient="records")


    def get_patterns_summary(self):
        overview = self.get_overview()

        demand = self.connection.execute("""
            SELECT
                pickup_hour_of_day AS group_value,
                SUM(total_trips) AS trip_count
            FROM historical_trips
            WHERE location_role = 'pickup'
            GROUP BY pickup_hour_of_day
            ORDER BY pickup_hour_of_day
        """).fetchdf().to_dict(orient="records")

        revenue = self.connection.execute("""
            SELECT
                borough AS group_value,
                SUM(valid_fare_trips) AS valid_fare_trips,
                SUM(fare_sum) AS fare_revenue,
                SUM(fare_sum) / NULLIF(SUM(valid_fare_trips), 0)
                    AS average_fare
            FROM historical_trips
            WHERE location_role = 'pickup'
            GROUP BY borough
            ORDER BY fare_revenue DESC
        """).fetchdf().to_dict(orient="records")

        metrics = self.connection.execute("""
            SELECT
                SUM(valid_fare_trips) AS valid_fare_trips,
                SUM(fare_sum) / NULLIF(SUM(valid_fare_trips), 0)
                    AS average_fare,
                SUM(distance_sum) / NULLIF(SUM(valid_distance_trips), 0)
                    AS average_distance_miles,
                SUM(duration_seconds_sum)
                    / NULLIF(SUM(valid_duration_trips), 0) / 60.0
                    AS average_duration_minutes
            FROM historical_trips
            WHERE location_role = 'pickup'
        """).fetchdf().to_dict(orient="records")

        first_date = overview["first_date"]
        last_date = overview["last_date"]

        return {
            "coverage": {
                "start_date": first_date.isoformat()
                    if hasattr(first_date, "isoformat") else str(first_date),
                "end_date": last_date.isoformat()
                    if hasattr(last_date, "isoformat") else str(last_date),
                "total_trips": int(overview["total_trips"] or 0),
            },
            "demand": {"data": demand},
            "revenue": {"data": revenue},
            "metrics": {"data": metrics},
        }

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

