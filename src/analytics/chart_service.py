
from datetime import date

import pandas as pd

from src.analytics.engine import MoveIQAnalytics


class MoveIQChartService:

    GROUP_OPTIONS = {
        "hour",
        "weekday",
        "borough",
        "pickup_zone",
        "dropoff_zone",
    }

    CHART_TYPES = {"bar", "line", "pie"}

    WEEKDAY_ORDER = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7,
    }

    METRIC_LABELS = {
        "trip_count": "Number of Trips",
        "fare_revenue": "Fare Revenue ($)",
        "average_fare": "Average Fare ($)",
        "valid_fare_trips": "Valid Fare Trips",
    }

    def __init__(self, analytics):
        self.analytics = analytics

    def _validate_request(self, request):

        if not isinstance(request, dict):
            raise ValueError("Chart request must be a dictionary.")

        metric = request.get("metric")
        group_by = request.get("group_by")
        chart_type = request.get("chart_type", "bar")

        if metric not in self.METRIC_LABELS:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Choose from {list(self.METRIC_LABELS)}"
            )

        if group_by not in self.GROUP_OPTIONS:
            raise ValueError(
                f"Unsupported grouping: {group_by}. "
                f"Choose from {sorted(self.GROUP_OPTIONS)}"
            )

        if chart_type not in self.CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        if chart_type == "line" and group_by not in {"hour", "weekday"}:
            raise ValueError(
                "Line charts are supported only for hour or weekday grouping."
            )

        if chart_type == "pie" and group_by == "hour":
            raise ValueError(
                "Pie charts by hour are not supported. Use a bar chart."
            )

        filters = request.get("filters", {})

        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary.")

        allowed_filters = {
            "start_date",
            "end_date",
            "weekdays",
            "hours",
            "borough",
            "pickup_zone_ids",
            "dropoff_zone_ids",
        }

        unknown = set(filters) - allowed_filters

        if unknown:
            raise ValueError(
                f"Unsupported filters: {sorted(unknown)}"
            )

        return metric, group_by, chart_type, filters

    def _clean_category(self, value, group_by):

        if pd.isna(value):
            return "Unknown"

        if group_by == "hour":
            return int(float(value))

        value = str(value).strip()

        if value.lower() in {"nan", "none", "null", ""}:
            return "Unknown"

        return value

    def build_chart(self, request):

        metric, group_by, chart_type, filters = (
            self._validate_request(request)
        )

        if metric == "trip_count":
            result = self.analytics.get_filtered_demand(
                group_by=group_by,
                **filters,
            )
            value_column = "trip_count"

        else:
            result = self.analytics.get_filtered_revenue(
                group_by=group_by,
                **filters,
            )
            value_column = metric

        totals = {}

        for _, row in result.iterrows():

            category = self._clean_category(
                row["group_value"], group_by
            )

            value = row[value_column]

            if pd.isna(value):
                continue

            if group_by == "hour":
                value = int(value)

            if metric == "average_fare":
                weight = row["valid_fare_trips"]

                if pd.isna(weight) or weight == 0:
                    continue

                if category not in totals:
                    totals[category] = {
                        "weighted_total": 0.0,
                        "weight": 0,
                    }

                totals[category]["weighted_total"] += (
                    float(value) * int(weight)
                )
                totals[category]["weight"] += int(weight)

            else:
                totals[category] = (
                    totals.get(category, 0) + value
                )

        data = []

        for category, value in totals.items():

            if metric == "average_fare":
                value = (
                    value["weighted_total"] / value["weight"]
                )

            data.append({
                "category": category,
                "value": value,
            })

        if group_by == "hour":
            data.sort(key=lambda item: item["category"])

        elif group_by == "weekday":
            data.sort(
                key=lambda item: self.WEEKDAY_ORDER.get(
                    item["category"], 99
                )
            )

        if not data:
            coverage = self.analytics.get_overview().iloc[0].to_dict()

            raise ValueError(
                "No chart data found. "
                f"Requested metric: {metric}. "
                f"Grouping: {group_by}. "
                f"Filters: {filters}. "
                f"Dataset coverage: {coverage}"
            )

        return {
            "chart_type": chart_type,
            "title": (
                f"{self.METRIC_LABELS[metric]} by "
                f"{group_by.replace('_', ' ').title()}"
            ),
            "x_key": "category",
            "series": [
                {
                    "data_key": "value",
                    "label": self.METRIC_LABELS[metric],
                }
            ],
            "data": data,
            "metadata": {
                "metric": metric,
                "group_by": group_by,
                "filters": filters,
                "record_count": len(data),
            },
        }

    def build_comparison_chart(
        self,
        segment_a,
        segment_b,
        metric="fare_revenue",
        name_a="Segment A",
        name_b="Segment B",
        chart_type="bar",
    ):

        if metric not in self.METRIC_LABELS:
            raise ValueError(
                f"Unsupported comparison metric: {metric}"
            )

        if chart_type not in {"bar", "line"}:
            raise ValueError(
                "Comparison charts support bar and line types."
            )

        result = self.analytics.compare_segments(
            segment_a=segment_a,
            segment_b=segment_b,
            name_a=name_a,
            name_b=name_b,
        )

        value_column = (
            "total_trips" if metric == "trip_count" else metric
        )

        data = []

        for _, row in result.iterrows():

            value = row[value_column]

            if pd.isna(value):
                continue

            data.append({
                "segment": str(row["segment"]),
                "value": value,
            })

        if not data:
            raise ValueError(
                "No data available for comparison chart."
            )

        return {
            "chart_type": chart_type,
            "title": f"{self.METRIC_LABELS[metric]} Comparison",
            "x_key": "segment",
            "series": [
                {
                    "data_key": "value",
                    "label": self.METRIC_LABELS[metric],
                }
            ],
            "data": data,
            "metadata": {
                "metric": metric,
                "comparison": True,
                "segments": [name_a, name_b],
            },
        }
