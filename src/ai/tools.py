import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.copilot_engine import HistoricalCopilotAnalytics
from src.analytics.chart_service import MoveIQChartService
from src.ml.extended_forecast_service import ExtendedForecastService


class MoveIQToolbox:
    """
    Controlled interface between the MoveIQ AI agent
    and its existing analytics and forecasting services.
    """

    def __init__(self, analytics=None, forecast=None):
        self.analytics = analytics or HistoricalCopilotAnalytics()
        self.forecast = forecast or ExtendedForecastService()
        self.chart_service = MoveIQChartService(self.analytics)

        self._overview = self.analytics.get_overview()
        row = self._overview.iloc[0]

        self.analytics_metadata = {
            "source": "NYC TLC historical zone-hourly aggregate",
            "start_date": str(row["first_pickup"]),
            "end_date": str(row["last_pickup"]),
            "total_trips": int(row["total_trips"]),
        }
    @staticmethod
    def _normalize(value):
        """Convert Pandas and NumPy values into JSON-safe values."""

        if isinstance(value, dict):
            return {
                str(k): MoveIQToolbox._normalize(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                MoveIQToolbox._normalize(v)
                for v in value
            ]

        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and not np.isfinite(value):
            return None

        if pd.isna(value):
            return None

        return value

    @classmethod
    def _records(cls, dataframe):
        records = dataframe.to_dict(orient="records")
        return cls._normalize(records)

    @classmethod
    def _response(cls, tool, source, coverage, data):
        result = {
            "tool": tool,
            "source": source,
            "coverage": coverage,
            "data": cls._normalize(data),
        }

        # Guarantee that the result can be serialized as strict JSON.
        return json.loads(
            json.dumps(result, allow_nan=False)
        )

    def get_dataset_overview(self):
        return self._response(
            tool="get_dataset_overview",
            source=self.analytics_metadata["source"],
            coverage={
                "start": self.analytics_metadata["start_date"],
                "end": self.analytics_metadata["end_date"],
            },
            data=self.analytics.get_overview().to_dict(
                orient="records"
            ),
        )

    def get_trip_metrics(self):
        return self._response(
            tool="get_trip_metrics",
            source=self.analytics_metadata["source"],
            coverage=self.analytics_metadata,
            data=self.analytics.get_trip_metrics().to_dict(
                orient="records"
            ),
        )

    def get_demand_analysis(
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
        result = self.analytics.get_filtered_demand(
            group_by=group_by,
            start_date=start_date,
            end_date=end_date,
            weekdays=weekdays,
            hours=hours,
            borough=borough,
            pickup_zone_ids=pickup_zone_ids,
            dropoff_zone_ids=dropoff_zone_ids,
        )

        return self._response(
            tool="get_demand_analysis",
            source=self.analytics_metadata["source"],
            coverage=self.analytics_metadata,
            data=self._records(result),
        )

    def get_revenue_analysis(
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
        result = self.analytics.get_filtered_revenue(
            group_by=group_by,
            start_date=start_date,
            end_date=end_date,
            weekdays=weekdays,
            hours=hours,
            borough=borough,
            pickup_zone_ids=pickup_zone_ids,
            dropoff_zone_ids=dropoff_zone_ids,
        )

        return self._response(
            tool="get_revenue_analysis",
            source=self.analytics_metadata["source"],
            coverage=self.analytics_metadata,
            data=self._records(result),
        )

    def get_zone_rankings(self, kind="pickup", limit=10):
        limit = int(limit)

        if not 1 <= limit <= 300:
            raise ValueError("Limit must be between 1 and 300.")

        if kind == "pickup":
            result = self.analytics.get_top_pickup_zones(limit)

        elif kind == "dropoff":
            result = self.analytics.get_top_dropoff_zones(limit)

        else:
            raise ValueError(
                "Kind must be 'pickup' or 'dropoff'."
            )

        return self._response(
            tool="get_zone_rankings",
            source=self.analytics_metadata["source"],
            coverage=self.analytics_metadata,
            data=self._records(result),
        )

    def compare_segments(
        self,
        segment_a,
        segment_b,
        name_a="Segment A",
        name_b="Segment B",
    ):
        if not isinstance(segment_a, dict) or not isinstance(segment_b, dict):
            raise ValueError("Both segments must be dictionaries.")

        result = self.analytics.compare_segments(
            segment_a=segment_a,
            segment_b=segment_b,
            name_a=name_a,
            name_b=name_b,
        )

        return self._response(
            tool="compare_segments",
            source=self.analytics_metadata["source"],
            coverage=self.analytics_metadata,
            data=self._records(result),
        )

    def get_citywide_forecast(self, as_of=None):
        result = self.forecast.forecast_citywide(as_of=as_of)

        origin = str(result["forecast_origin"].iloc[0])

        return self._response(
            tool="get_citywide_forecast",
            source="MoveIQ extended citywide demand forecasting model",
            coverage={
                "forecast_origin": origin,
                "forecast_start": str(result["target_timestamp"].min()),
                "forecast_end": str(result["target_timestamp"].max()),
                "forecast_horizon_hours": 24,
                "timezone": "America/New_York",
                "timestamp_semantics": "Naive timestamps represent New York local time, not UTC.",
            },
            data=self._records(result),
        )

    def get_zone_forecast(self, zone_ids, as_of=None):
        if isinstance(zone_ids, int):
            zone_ids = [zone_ids]

        if not isinstance(zone_ids, (list, tuple)) or not zone_ids:
            raise ValueError("Provide at least one taxi zone ID.")

        if len(zone_ids) > 263:
            raise ValueError("A maximum of 263 zones can be requested.")

        zone_ids = [int(zone_id) for zone_id in zone_ids]

        if len(set(zone_ids)) != len(zone_ids):
            raise ValueError("Duplicate zone IDs are not allowed.")

        result = self.forecast.forecast_zones(
            zone_ids=zone_ids,
            as_of=as_of,
        )

        origin = str(result["forecast_origin"].iloc[0])

        return self._response(
            tool="get_zone_forecast",
            source="MoveIQ extended zone demand forecasting model",
            coverage={
                "forecast_origin": origin,
                "forecast_start": str(result["target_timestamp"].min()),
                "forecast_end": str(result["target_timestamp"].max()),
                "forecast_horizon_hours": 24,
                "timezone": "America/New_York",
                "timestamp_semantics": "Naive timestamps represent New York local time, not UTC.",
                "requested_zone_ids": zone_ids,
            },
            data=self._records(result),
        )

    def get_borough_forecast(self, borough, as_of=None):
        if not isinstance(borough, str) or not borough.strip():
            raise ValueError("A borough name is required.")

        result = self.forecast.forecast_borough(
            borough=borough.strip(),
            as_of=as_of,
        )

        origin = str(result["forecast_origin"].iloc[0])

        return self._response(
            tool="get_borough_forecast",
            source="MoveIQ extended zone demand forecasting model",
            coverage={
                "forecast_origin": origin,
                "forecast_start": str(result["target_timestamp"].min()),
                "forecast_end": str(result["target_timestamp"].max()),
                "forecast_horizon_hours": 24,
                "timezone": "America/New_York",
                "timestamp_semantics": "Naive timestamps represent New York local time, not UTC.",
                "borough": borough,
            },
            data=self._records(result),
        )

    def get_forecast_insights(
        self,
        scope="citywide",
        borough=None,
        zone_ids=None,
        as_of=None,
    ):
        """Return deterministically ranked hourly forecast insights."""

        if scope == "citywide":
            forecast = self.get_citywide_forecast(as_of=as_of)

        elif scope == "borough":
            if not borough:
                raise ValueError("A borough is required.")

            forecast = self.get_borough_forecast(
                borough=borough,
                as_of=as_of,
            )

        elif scope == "zones":
            if not zone_ids:
                raise ValueError("At least one zone ID is required.")

            forecast = self.get_zone_forecast(
                zone_ids=zone_ids,
                as_of=as_of,
            )

        else:
            raise ValueError("scope must be citywide, borough, or zones.")

        records = forecast["data"]

        if not records:
            raise ValueError("The forecast returned no records.")

        # Aggregate selected zones by forecast hour.
        hourly = {}

        for record in records:
            timestamp = record["target_timestamp"]
            demand = float(record["predicted_demand"])

            if not np.isfinite(demand):
                raise ValueError("Forecast contains a non-finite prediction.")

            hourly[timestamp] = hourly.get(timestamp, 0.0) + demand

        hourly_records = [
            {
                "target_timestamp": timestamp,
                "predicted_demand": round(demand, 2),
            }
            for timestamp, demand in hourly.items()
        ]

        busiest = sorted(
            hourly_records,
            key=lambda row: (
                -row["predicted_demand"],
                row["target_timestamp"],
            ),
        )[:3]

        quietest = sorted(
            hourly_records,
            key=lambda row: (
                row["predicted_demand"],
                row["target_timestamp"],
            ),
        )[:3]

        busiest = [
            {"rank": i + 1, **row}
            for i, row in enumerate(busiest)
        ]

        quietest = [
            {"rank": i + 1, **row}
            for i, row in enumerate(quietest)
        ]

        coverage = dict(forecast["coverage"])
        coverage["timezone"] = "America/New_York"
        coverage["timestamp_semantics"] = (
            "Forecast timestamps represent New York local time, not UTC."
        )

        return self._response(
            tool="get_forecast_insights",
            source=forecast["source"],
            coverage=coverage,
            data={
                "scope": scope,
                "top_3_demand_hours": busiest,
                "lowest_3_demand_hours": quietest,
                "hourly_forecast": sorted(
                    hourly_records,
                    key=lambda row: row["target_timestamp"],
                ),
            },
        )

    def generate_chart(self, request):
        result = self.chart_service.build_chart(request)

        return self._response(
            tool="generate_chart",
            source=self.analytics_metadata["source"],
            coverage=self.analytics_metadata,
            data=result,
        )

    def close(self):
        if hasattr(self.analytics, "close"):
            self.analytics.close()


