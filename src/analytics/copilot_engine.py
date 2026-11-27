from pathlib import Path

import duckdb
import pandas as pd


class HistoricalCopilotAnalytics:
    """
    DuckDB compatibility layer for Copilot.

    Reads the pre-aggregated NYC TLC dataset covering
    2024-01-01 through 2026-09-30.
    """

    DEFAULT_DATASET = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "analytics"
        / "historical"
        / "zone_hourly_2024_2026.parquet"
    )

    GROUP_COLUMNS = {
        "hour": ("pickup_hour_of_day", "Hour"),
        "weekday": ("dayname(pickup_date)", "Weekday"),
        "borough": ("borough", "Borough"),
        "pickup_zone": ("zone_name", "Zone"),
        "dropoff_zone": ("zone_name", "Zone"),
    }

    def __init__(self, dataset_path=None):
        self.dataset_path = Path(dataset_path or self.DEFAULT_DATASET)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Historical analytics dataset not found: {self.dataset_path}"
            )

        self.connection = duckdb.connect(database=":memory:")
        self.connection.execute("SET memory_limit = '256MB'")
        self.connection.execute("SET threads = 1")

        escaped_path = str(self.dataset_path.resolve()).replace("\\", "/").replace("'", "''")

        self.connection.execute(
            f"""
            CREATE VIEW historical_trips AS
            SELECT *
            FROM read_parquet('{escaped_path}')
            """
        )

        self._coverage = self.connection.execute(
            """
            SELECT
                MIN(pickup_date) AS first_date,
                MAX(pickup_date) AS last_date,
                COUNT(DISTINCT location_id) AS zone_count,
                SUM(total_trips) AS total_trips,
                COUNT(*) AS aggregate_rows
            FROM historical_trips
            WHERE location_role = 'pickup'
            """
        ).df().iloc[0].to_dict()

    @staticmethod
    def _validate_group(group_by):
        if group_by not in HistoricalCopilotAnalytics.GROUP_COLUMNS:
            raise ValueError(
                "group_by must be hour, weekday, borough, pickup_zone, or dropoff_zone."
            )

    @staticmethod
    def _validate_zone_filters(pickup_zone_ids, dropoff_zone_ids, group_by):
        if pickup_zone_ids and dropoff_zone_ids:
            raise ValueError(
                "The historical aggregate cannot filter pickup and dropoff "
                "zone IDs together because it does not retain individual "
                "pickup/dropoff trip pairs."
            )

        if pickup_zone_ids and group_by == "dropoff_zone":
            raise ValueError(
                "Pickup-zone filters cannot be combined with dropoff-zone grouping."
            )

        if dropoff_zone_ids and group_by == "pickup_zone":
            raise ValueError(
                "Dropoff-zone filters cannot be combined with pickup-zone grouping."
            )

    def _build_filters(
        self,
        start_date=None,
        end_date=None,
        weekdays=None,
        hours=None,
        borough=None,
        pickup_zone_ids=None,
        dropoff_zone_ids=None,
        group_by="hour",
    ):
        self._validate_group(group_by)
        self._validate_zone_filters(
            pickup_zone_ids,
            dropoff_zone_ids,
            group_by,
        )

        role = "dropoff" if group_by == "dropoff_zone" else "pickup"

        if dropoff_zone_ids:
            role = "dropoff"

        conditions = ["location_role = ?"]
        params = [role]

        if start_date:
            conditions.append("pickup_date >= CAST(? AS DATE)")
            params.append(str(start_date))

        if end_date:
            conditions.append("pickup_date <= CAST(? AS DATE)")
            params.append(str(end_date))

        if weekdays:
            values = [str(value).strip() for value in weekdays]
            placeholders = ", ".join(["?"] * len(values))
            conditions.append(f"dayname(pickup_date) IN ({placeholders})")
            params.extend(values)

        if hours:
            values = [int(value) for value in hours]

            if any(value < 0 or value > 23 for value in values):
                raise ValueError("Hours must be between 0 and 23.")

            placeholders = ", ".join(["?"] * len(values))
            conditions.append(
                f"pickup_hour_of_day IN ({placeholders})"
            )
            params.extend(values)

        if borough:
            conditions.append("borough = ?")
            params.append(str(borough))

        zone_ids = pickup_zone_ids or dropoff_zone_ids

        if zone_ids:
            values = [int(value) for value in zone_ids]
            placeholders = ", ".join(["?"] * len(values))
            conditions.append(f"location_id IN ({placeholders})")
            params.extend(values)

        return " AND ".join(conditions), params

    def _query(self, sql, params=None):
        return self.connection.execute(sql, params or []).df()

    def get_overview(self):
        row = self._coverage

        return pd.DataFrame([{
            "first_pickup": row["first_date"],
            "last_pickup": row["last_date"],
            "zone_count": int(row["zone_count"]),
            "total_trips": int(row["total_trips"]),
            "aggregate_rows": int(row["aggregate_rows"]),
        }])

    def get_trip_metrics(self):
        return self._query(
            """
            SELECT
                SUM(total_trips) AS total_trips,
                SUM(valid_fare_trips) AS valid_fare_trips,
                SUM(fare_sum) AS fare_revenue,
                SUM(fare_sum) / NULLIF(SUM(valid_fare_trips), 0) AS average_fare,
                SUM(valid_distance_trips) AS valid_distance_trips,
                SUM(distance_sum) / NULLIF(SUM(valid_distance_trips), 0)
                    AS average_distance_miles,
                SUM(valid_duration_trips) AS valid_duration_trips,
                SUM(duration_seconds_sum)
                    / NULLIF(SUM(valid_duration_trips), 0) / 60
                    AS average_duration_minutes
            FROM historical_trips
            WHERE location_role = 'pickup'
            """
        )

    def get_filtered_demand(
        self,
        group_by="hour",
        start_date=None,
        end_date=None,
        weekdays=None,
        hours=None,
        borough=None,
        pickup_zone_ids=None,
        dropoff_zone_ids=None,
    ):
        self._validate_group(group_by)

        where_sql, params = self._build_filters(
            start_date=start_date,
            end_date=end_date,
            weekdays=weekdays,
            hours=hours,
            borough=borough,
            pickup_zone_ids=pickup_zone_ids,
            dropoff_zone_ids=dropoff_zone_ids,
            group_by=group_by,
        )

        group_expression, _ = self.GROUP_COLUMNS[group_by]

        return self._query(
            f"""
            SELECT
                CAST({group_expression} AS VARCHAR) AS group_value,
                SUM(total_trips) AS trip_count
            FROM historical_trips
            WHERE {where_sql}
            GROUP BY 1
            ORDER BY trip_count DESC
            """,
            params,
        )

    def get_filtered_revenue(
        self,
        group_by="borough",
        start_date=None,
        end_date=None,
        weekdays=None,
        hours=None,
        borough=None,
        pickup_zone_ids=None,
        dropoff_zone_ids=None,
    ):
        self._validate_group(group_by)

        where_sql, params = self._build_filters(
            start_date=start_date,
            end_date=end_date,
            weekdays=weekdays,
            hours=hours,
            borough=borough,
            pickup_zone_ids=pickup_zone_ids,
            dropoff_zone_ids=dropoff_zone_ids,
            group_by=group_by,
        )

        group_expression, _ = self.GROUP_COLUMNS[group_by]

        return self._query(
            f"""
            SELECT
                CAST({group_expression} AS VARCHAR) AS group_value,
                SUM(valid_fare_trips) AS valid_fare_trips,
                SUM(fare_sum) AS fare_revenue,
                SUM(fare_sum) / NULLIF(SUM(valid_fare_trips), 0)
                    AS average_fare
            FROM historical_trips
            WHERE {where_sql}
            GROUP BY 1
            ORDER BY fare_revenue DESC
            """,
            params,
        )

    def get_top_pickup_zones(self, limit=10):
        return self._top_zones("pickup", limit)

    def get_top_dropoff_zones(self, limit=10):
        return self._top_zones("dropoff", limit)

    def _top_zones(self, role, limit):
        if role not in ("pickup", "dropoff"):
            raise ValueError("Invalid location role.")

        limit = int(limit)

        if not 1 <= limit <= 300:
            raise ValueError("Limit must be between 1 and 300.")

        return self._query(
            """
            SELECT
                location_id,
                zone_name,
                borough,
                SUM(total_trips) AS trip_count
            FROM historical_trips
            WHERE location_role = ?
            GROUP BY location_id, zone_name, borough
            ORDER BY trip_count DESC
            LIMIT ?
            """,
            [role, limit],
        )

    def compare_segments(
        self,
        segment_a,
        segment_b,
        name_a="Segment A",
        name_b="Segment B",
    ):
        results = []

        for name, segment in (
            (name_a, segment_a),
            (name_b, segment_b),
        ):
            if not isinstance(segment, dict):
                raise ValueError("Each segment must be a dictionary.")

            allowed = {
                "start_date",
                "end_date",
                "weekdays",
                "hours",
                "borough",
                "pickup_zone_ids",
                "dropoff_zone_ids",
            }

            unknown = set(segment) - allowed

            if unknown:
                raise ValueError(
                    f"Unsupported segment filters: {', '.join(sorted(unknown))}"
                )

            demand = self.get_filtered_demand(
                group_by="hour",
                **segment,
            )

            revenue = self.get_filtered_revenue(
                group_by="hour",
                **segment,
            )

            results.append({
                "segment": name,
                "total_trips": int(demand["trip_count"].sum()),
                "valid_fare_trips": int(revenue["valid_fare_trips"].sum()),
                "fare_revenue": float(revenue["fare_revenue"].sum()),
                "average_fare": (
                    float(revenue["fare_revenue"].sum())
                    / int(revenue["valid_fare_trips"].sum())
                    if int(revenue["valid_fare_trips"].sum()) else None
                ),
            })

        return pd.DataFrame(results)

    def close(self):
        if self.connection:
            self.connection.close()
