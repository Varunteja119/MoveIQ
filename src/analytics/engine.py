from pathlib import Path

import duckdb
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yellow_tripdata_2025-01_cleaned.parquet"
)

DEFAULT_ZONE_LOOKUP = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "taxi_zone_lookup.csv"
)


# --------------------------------------------------
# MOVEIQ ANALYTICS ENGINE
# --------------------------------------------------

class MoveIQAnalytics:

    def __init__(
        self,
        dataset_path=DEFAULT_DATASET,
        zone_lookup_path=DEFAULT_ZONE_LOOKUP,
    ):

        self.dataset_path = Path(dataset_path)
        self.zone_lookup_path = Path(zone_lookup_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.connection = duckdb.connect(database=":memory:")

        # Query Parquet directly without importing all records
        self.connection.from_parquet(
            str(self.dataset_path)
        ).create_view("taxi_trips")

        # Register neighborhood lookup if available
        if self.zone_lookup_path.exists():

            zones = pd.read_csv(self.zone_lookup_path)

            zones = zones.rename(columns={
                "LocationID": "location_id",
                "Zone": "zone_name",
                "Borough": "borough",
                "service_zone": "service_zone",
            })

            self.connection.register(
                "zone_lookup_data",
                zones,
            )

            self.connection.execute("""
                CREATE VIEW zone_lookup AS
                SELECT *
                FROM zone_lookup_data
            """)

        print("MoveIQ Analytics Engine initialized.")

    # --------------------------------------------------
    # 1. DATASET OVERVIEW
    # --------------------------------------------------

    def get_overview(self):

        return self.connection.execute("""
            SELECT
                COUNT(*) AS total_trips,
                MIN(tpep_pickup_datetime) AS first_pickup,
                MAX(tpep_pickup_datetime) AS last_pickup,
                COUNT(DISTINCT PULocationID) AS pickup_zones,
                COUNT(DISTINCT DOLocationID) AS dropoff_zones,
                COUNT(*) FILTER (
                    WHERE is_valid_fare
                ) AS valid_fare_records,
                COUNT(*) FILTER (
                    WHERE is_valid_distance
                ) AS valid_distance_records,
                COUNT(*) FILTER (
                    WHERE is_valid_duration
                ) AS valid_duration_records
            FROM taxi_trips
        """).df()

    # --------------------------------------------------
    # 2. HOURLY DEMAND
    # --------------------------------------------------

    def get_hourly_demand(self):

        return self.connection.execute("""
            SELECT
                EXTRACT(HOUR FROM tpep_pickup_datetime)::INTEGER
                    AS pickup_hour,
                COUNT(*) AS trip_count
            FROM taxi_trips
            GROUP BY pickup_hour
            ORDER BY pickup_hour
        """).df()

    # --------------------------------------------------
    # 3. WEEKDAY DEMAND
    # --------------------------------------------------

    def get_weekday_demand(self):

        return self.connection.execute("""
            SELECT
                CASE
                    WHEN ISODOW(tpep_pickup_datetime) = 1 THEN 'Monday'
                    WHEN ISODOW(tpep_pickup_datetime) = 2 THEN 'Tuesday'
                    WHEN ISODOW(tpep_pickup_datetime) = 3 THEN 'Wednesday'
                    WHEN ISODOW(tpep_pickup_datetime) = 4 THEN 'Thursday'
                    WHEN ISODOW(tpep_pickup_datetime) = 5 THEN 'Friday'
                    WHEN ISODOW(tpep_pickup_datetime) = 6 THEN 'Saturday'
                    WHEN ISODOW(tpep_pickup_datetime) = 7 THEN 'Sunday'
                END AS weekday,
                COUNT(*) AS trip_count
            FROM taxi_trips
            GROUP BY ISODOW(tpep_pickup_datetime)
            ORDER BY ISODOW(tpep_pickup_datetime)
        """).df()

    # --------------------------------------------------
    # 4. TOP PICKUP NEIGHBORHOODS
    # --------------------------------------------------

    def get_top_pickup_zones(self, limit=10):

        limit = max(1, min(int(limit), 300))

        return self.connection.execute("""
            SELECT
                t.PULocationID AS location_id,
                z.zone_name,
                z.borough,
                COUNT(*) AS trip_count
            FROM taxi_trips t
            LEFT JOIN zone_lookup z
                ON t.PULocationID = z.location_id
            GROUP BY
                t.PULocationID,
                z.zone_name,
                z.borough
            ORDER BY trip_count DESC
            LIMIT ?
        """, [limit]).df()

    # --------------------------------------------------
    # 5. TOP DROPOFF NEIGHBORHOODS
    # --------------------------------------------------

    def get_top_dropoff_zones(self, limit=10):

        limit = max(1, min(int(limit), 300))

        return self.connection.execute("""
            SELECT
                t.DOLocationID AS location_id,
                z.zone_name,
                z.borough,
                COUNT(*) AS trip_count
            FROM taxi_trips t
            LEFT JOIN zone_lookup z
                ON t.DOLocationID = z.location_id
            GROUP BY
                t.DOLocationID,
                z.zone_name,
                z.borough
            ORDER BY trip_count DESC
            LIMIT ?
        """, [limit]).df()

    # --------------------------------------------------
    # 6. TRIP METRICS
    # --------------------------------------------------

    def get_trip_metrics(self):

        return self.connection.execute("""
            SELECT
                COUNT(*) FILTER (
                    WHERE is_valid_fare
                ) AS valid_fare_trips,

                ROUND(AVG(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2)
                    AS average_fare,

                ROUND(AVG(cleaned_trip_distance)
                    FILTER (WHERE is_valid_distance), 2)
                    AS average_distance_miles,

                ROUND(AVG(trip_duration_minutes)
                    FILTER (WHERE is_valid_duration), 2)
                    AS average_duration_minutes,

                ROUND(MEDIAN(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2)
                    AS median_fare,

                ROUND(MEDIAN(cleaned_trip_distance)
                    FILTER (WHERE is_valid_distance), 2)
                    AS median_distance_miles

            FROM taxi_trips
        """).df()

    # --------------------------------------------------
    # 7. PAYMENT DISTRIBUTION
    # --------------------------------------------------

    def get_payment_distribution(self):

        return self.connection.execute("""
            SELECT
                CASE payment_type
                    WHEN 0 THEN 'Flex Fare'
                    WHEN 1 THEN 'Credit card'
                    WHEN 2 THEN 'Cash'
                    WHEN 3 THEN 'No charge'
                    WHEN 4 THEN 'Dispute'
                    WHEN 5 THEN 'Unknown'
                    WHEN 6 THEN 'Voided trip'
                    ELSE 'Unrecognized'
                END AS payment_method,
                COUNT(*) AS trip_count
            FROM taxi_trips
            GROUP BY payment_method
            ORDER BY trip_count DESC
        """).df()

    # --------------------------------------------------
    # 8. REVENUE METRICS
    # --------------------------------------------------

    def get_revenue_metrics(self):

        return self.connection.execute("""
            SELECT
                COUNT(*) FILTER (
                    WHERE is_valid_fare
                ) AS valid_fare_trips,

                ROUND(
                    SUM(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2
                ) AS total_fare_revenue,

                ROUND(
                    AVG(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2
                ) AS average_fare_per_trip,

                ROUND(
                    MEDIAN(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2
                ) AS median_fare_per_trip

            FROM taxi_trips
        """).df()

    # --------------------------------------------------
    # 9. HOURLY REVENUE
    # --------------------------------------------------

    def get_hourly_revenue(self):

        return self.connection.execute("""
            SELECT
                EXTRACT(
                    HOUR FROM tpep_pickup_datetime
                )::INTEGER AS pickup_hour,

                COUNT(*) FILTER (
                    WHERE is_valid_fare
                ) AS valid_fare_trips,

                ROUND(
                    SUM(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2
                ) AS fare_revenue,

                ROUND(
                    AVG(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2
                ) AS average_fare

            FROM taxi_trips
            GROUP BY pickup_hour
            ORDER BY pickup_hour
        """).df()

    # --------------------------------------------------
    # 10. WEEKDAY REVENUE
    # --------------------------------------------------

    def get_weekday_revenue(self):

        return self.connection.execute("""
            SELECT
                CASE
                    WHEN ISODOW(tpep_pickup_datetime) = 1 THEN 'Monday'
                    WHEN ISODOW(tpep_pickup_datetime) = 2 THEN 'Tuesday'
                    WHEN ISODOW(tpep_pickup_datetime) = 3 THEN 'Wednesday'
                    WHEN ISODOW(tpep_pickup_datetime) = 4 THEN 'Thursday'
                    WHEN ISODOW(tpep_pickup_datetime) = 5 THEN 'Friday'
                    WHEN ISODOW(tpep_pickup_datetime) = 6 THEN 'Saturday'
                    WHEN ISODOW(tpep_pickup_datetime) = 7 THEN 'Sunday'
                END AS weekday,

                COUNT(*) FILTER (
                    WHERE is_valid_fare
                ) AS valid_fare_trips,

                ROUND(
                    SUM(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2
                ) AS fare_revenue,

                ROUND(
                    AVG(cleaned_fare_amount)
                    FILTER (WHERE is_valid_fare), 2
                ) AS average_fare

            FROM taxi_trips
            GROUP BY ISODOW(tpep_pickup_datetime)
            ORDER BY ISODOW(tpep_pickup_datetime)
        """).df()

    # --------------------------------------------------
    # 11. REVENUE BY PICKUP NEIGHBORHOOD
    # --------------------------------------------------

    def get_zone_revenue(self, limit=10):

        limit = max(1, min(int(limit), 300))

        return self.connection.execute("""
            SELECT
                t.PULocationID AS location_id,
                z.zone_name,
                z.borough,

                COUNT(*) FILTER (
                    WHERE t.is_valid_fare
                ) AS valid_fare_trips,

                ROUND(
                    SUM(t.cleaned_fare_amount)
                    FILTER (WHERE t.is_valid_fare), 2
                ) AS fare_revenue,

                ROUND(
                    AVG(t.cleaned_fare_amount)
                    FILTER (WHERE t.is_valid_fare), 2
                ) AS average_fare

            FROM taxi_trips t

            LEFT JOIN zone_lookup z
                ON t.PULocationID = z.location_id

            GROUP BY
                t.PULocationID,
                z.zone_name,
                z.borough

            ORDER BY fare_revenue DESC
            LIMIT ?
        """, [limit]).df()

    # --------------------------------------------------
    # 12. DEMAND BY NEIGHBORHOOD AND HOUR
    # --------------------------------------------------

    def get_zone_hour_demand(self, limit=15):

        limit = max(1, min(int(limit), 300))

        return self.connection.execute("""
            WITH top_zones AS (
                SELECT
                    PULocationID,
                    COUNT(*) AS total_trips
                FROM taxi_trips
                GROUP BY PULocationID
                ORDER BY total_trips DESC
                LIMIT ?
            )

            SELECT
                t.PULocationID AS location_id,
                z.zone_name,
                z.borough,
                EXTRACT(
                    HOUR FROM t.tpep_pickup_datetime
                )::INTEGER AS pickup_hour,
                COUNT(*) AS trip_count

            FROM taxi_trips t

            INNER JOIN top_zones tz
                ON t.PULocationID = tz.PULocationID

            LEFT JOIN zone_lookup z
                ON t.PULocationID = z.location_id

            GROUP BY
                t.PULocationID,
                z.zone_name,
                z.borough,
                pickup_hour

            ORDER BY
                location_id,
                pickup_hour
        """, [limit]).df()

    # --------------------------------------------------
    # 13. SHARED DYNAMIC FILTER BUILDER
    # --------------------------------------------------

    def _build_filters(
        self,
        start_date=None,
        end_date=None,
        weekdays=None,
        hours=None,
        borough=None,
        pickup_zone_ids=None,
        dropoff_zone_ids=None,
    ):

        conditions = []
        params = []

        weekday_map = {
            "monday": 1,
            "tuesday": 2,
            "wednesday": 3,
            "thursday": 4,
            "friday": 5,
            "saturday": 6,
            "sunday": 7,
        }

        if start_date is not None:
            conditions.append(
                "CAST(t.tpep_pickup_datetime AS DATE) >= CAST(? AS DATE)"
            )
            params.append(start_date)

        if end_date is not None:
            conditions.append(
                "CAST(t.tpep_pickup_datetime AS DATE) <= CAST(? AS DATE)"
            )
            params.append(end_date)

        if start_date is not None and end_date is not None:
            if str(start_date) > str(end_date):
                raise ValueError("start_date cannot be after end_date")

        if weekdays is not None:
            if isinstance(weekdays, str):
                weekdays = [weekdays]

            weekday_numbers = []

            for day in weekdays:
                day_key = str(day).strip().lower()

                if day_key not in weekday_map:
                    raise ValueError(f"Invalid weekday: {day}")

                weekday_numbers.append(weekday_map[day_key])

            if weekday_numbers:
                placeholders = ", ".join(
                    ["?"] * len(weekday_numbers)
                )
                conditions.append(
                    f"ISODOW(t.tpep_pickup_datetime) IN ({placeholders})"
                )
                params.extend(weekday_numbers)

        if hours is not None:
            if isinstance(hours, int):
                hours = [hours]

            hours = [int(hour) for hour in hours]

            if any(hour < 0 or hour > 23 for hour in hours):
                raise ValueError("Hours must be between 0 and 23")

            if hours:
                placeholders = ", ".join(["?"] * len(hours))
                conditions.append(
                    f"EXTRACT(HOUR FROM t.tpep_pickup_datetime)::INTEGER "
                    f"IN ({placeholders})"
                )
                params.extend(hours)

        if borough is not None:
            if isinstance(borough, str):
                borough = [borough]

            borough = [str(value).strip() for value in borough]

            if borough:
                placeholders = ", ".join(["?"] * len(borough))
                conditions.append(
                    f"LOWER(z.borough) IN ({placeholders})"
                )
                params.extend([value.lower() for value in borough])

        if pickup_zone_ids is not None:
            if isinstance(pickup_zone_ids, int):
                pickup_zone_ids = [pickup_zone_ids]

            pickup_zone_ids = [int(value) for value in pickup_zone_ids]

            if pickup_zone_ids:
                placeholders = ", ".join(["?"] * len(pickup_zone_ids))
                conditions.append(
                    f"t.PULocationID IN ({placeholders})"
                )
                params.extend(pickup_zone_ids)

        if dropoff_zone_ids is not None:
            if isinstance(dropoff_zone_ids, int):
                dropoff_zone_ids = [dropoff_zone_ids]

            dropoff_zone_ids = [int(value) for value in dropoff_zone_ids]

            if dropoff_zone_ids:
                placeholders = ", ".join(["?"] * len(dropoff_zone_ids))
                conditions.append(
                    f"t.DOLocationID IN ({placeholders})"
                )
                params.extend(dropoff_zone_ids)

        where_clause = ""

        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        return where_clause, params

    # --------------------------------------------------
    # 14. FILTERED DEMAND
    # --------------------------------------------------

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

        group_options = {
            "hour": """
                EXTRACT(
                    HOUR FROM t.tpep_pickup_datetime
                )::INTEGER
            """,

            "weekday": """
                STRFTIME(t.tpep_pickup_datetime, '%A')
            """,

            "borough": "z.borough",

            "pickup_zone": """
                CAST(t.PULocationID AS VARCHAR)
                || ' - ' || COALESCE(z.zone_name, 'Unknown')
            """,

            "dropoff_zone": """
                CAST(t.DOLocationID AS VARCHAR)
                || ' - ' || COALESCE(dz.zone_name, 'Unknown')
            """,
        }

        if group_by not in group_options:
            raise ValueError(
                f"Unsupported group_by: {group_by}. "
                f"Choose from {list(group_options)}"
            )

        group_expression = group_options[group_by]

        where_clause, params = self._build_filters(
            start_date=start_date,
            end_date=end_date,
            weekdays=weekdays,
            hours=hours,
            borough=borough,
            pickup_zone_ids=pickup_zone_ids,
            dropoff_zone_ids=dropoff_zone_ids,
        )

        query = f"""
            SELECT
                {group_expression} AS group_value,
                COUNT(*) AS trip_count

            FROM taxi_trips t

            LEFT JOIN zone_lookup z
                ON t.PULocationID = z.location_id

            LEFT JOIN zone_lookup dz
                ON t.DOLocationID = dz.location_id

            {where_clause}

            GROUP BY group_value
            ORDER BY trip_count DESC
        """

        return self.connection.execute(query, params).df()

    # --------------------------------------------------
    # 15. FILTERED REVENUE
    # --------------------------------------------------

    def get_filtered_revenue(
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

        group_options = {
            "hour": """
                EXTRACT(
                    HOUR FROM t.tpep_pickup_datetime
                )::INTEGER
            """,

            "weekday": """
                STRFTIME(t.tpep_pickup_datetime, '%A')
            """,

            "borough": "z.borough",

            "pickup_zone": """
                CAST(t.PULocationID AS VARCHAR)
                || ' - ' || COALESCE(z.zone_name, 'Unknown')
            """,

            "dropoff_zone": """
                CAST(t.DOLocationID AS VARCHAR)
                || ' - ' || COALESCE(dz.zone_name, 'Unknown')
            """,
        }

        if group_by not in group_options:
            raise ValueError(
                f"Unsupported group_by: {group_by}. "
                f"Choose from {list(group_options)}"
            )

        group_expression = group_options[group_by]

        where_clause, params = self._build_filters(
            start_date=start_date,
            end_date=end_date,
            weekdays=weekdays,
            hours=hours,
            borough=borough,
            pickup_zone_ids=pickup_zone_ids,
            dropoff_zone_ids=dropoff_zone_ids,
        )

        query = f"""
            SELECT
                {group_expression} AS group_value,

                COUNT(*) FILTER (
                    WHERE t.is_valid_fare
                ) AS valid_fare_trips,

                ROUND(
                    SUM(t.cleaned_fare_amount)
                    FILTER (WHERE t.is_valid_fare), 2
                ) AS fare_revenue,

                ROUND(
                    AVG(t.cleaned_fare_amount)
                    FILTER (WHERE t.is_valid_fare), 2
                ) AS average_fare

            FROM taxi_trips t

            LEFT JOIN zone_lookup z
                ON t.PULocationID = z.location_id

            LEFT JOIN zone_lookup dz
                ON t.DOLocationID = dz.location_id

            {where_clause}

            GROUP BY group_value
            ORDER BY fare_revenue DESC
        """

        return self.connection.execute(query, params).df()

    # --------------------------------------------------
    # 16. COMPARE TWO SEGMENTS
    # --------------------------------------------------

    def compare_segments(
        self,
        segment_a,
        segment_b,
        name_a="Segment A",
        name_b="Segment B",
    ):

        allowed_filters = {
            "start_date",
            "end_date",
            "weekdays",
            "hours",
            "borough",
            "pickup_zone_ids",
            "dropoff_zone_ids",
        }

        results = []

        for name, filters in [
            (name_a, segment_a),
            (name_b, segment_b),
        ]:

            if not isinstance(filters, dict):
                raise ValueError("Each segment must be a dictionary.")

            unknown = set(filters) - allowed_filters

            if unknown:
                raise ValueError(
                    f"Unsupported filters: {sorted(unknown)}"
                )

            where_clause, params = self._build_filters(**filters)

            query = f"""
                SELECT
                    COUNT(*) AS total_trips,

                    COUNT(*) FILTER (
                        WHERE t.is_valid_fare
                    ) AS valid_fare_trips,

                    ROUND(
                        SUM(t.cleaned_fare_amount)
                        FILTER (WHERE t.is_valid_fare), 2
                    ) AS fare_revenue,

                    ROUND(
                        AVG(t.cleaned_fare_amount)
                        FILTER (WHERE t.is_valid_fare), 2
                    ) AS average_fare,

                    ROUND(
                        AVG(t.cleaned_trip_distance)
                        FILTER (WHERE t.is_valid_distance), 2
                    ) AS average_distance_miles,

                    ROUND(
                        AVG(t.trip_duration_minutes)
                        FILTER (WHERE t.is_valid_duration), 2
                    ) AS average_duration_minutes

                FROM taxi_trips t

                LEFT JOIN zone_lookup z
                    ON t.PULocationID = z.location_id

                {where_clause}
            """

            result = self.connection.execute(
                query, params
            ).df()

            result.insert(0, "segment", name)
            results.append(result)

        import pandas as pd

        return pd.concat(results, ignore_index=True)

    # --------------------------------------------------
    # CLOSE CONNECTION
    # --------------------------------------------------

    def close(self):
        self.connection.close()
