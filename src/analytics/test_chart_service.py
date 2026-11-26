import json

from src.analytics.engine import MoveIQAnalytics
from src.analytics.chart_service import MoveIQChartService


analytics = MoveIQAnalytics()
charts = MoveIQChartService(analytics)

try:
    print("\n" + "=" * 65)
    print("MOVEIQ — CHART SERVICE TESTS")
    print("=" * 65)

    print("\n1. HOURLY DEMAND BAR CHART")

    chart = charts.build_chart({
        "metric": "trip_count",
        "group_by": "hour",
        "chart_type": "bar",
    })

    print(json.dumps(chart, indent=2, default=str))

    print("\n2. MANHATTAN REVENUE BY HOUR")

    chart = charts.build_chart({
        "metric": "fare_revenue",
        "group_by": "hour",
        "chart_type": "line",
        "filters": {
            "borough": "Manhattan",
        },
    })

    print(json.dumps(chart, indent=2, default=str))

    print("\n3. REVENUE BY BOROUGH PIE CHART")

    chart = charts.build_chart({
        "metric": "fare_revenue",
        "group_by": "borough",
        "chart_type": "pie",
    })

    print(json.dumps(chart, indent=2, default=str))

    print("\n4. JFK VS LAGUARDIA COMPARISON")

    chart = charts.build_comparison_chart(
        segment_a={"pickup_zone_ids": 132},
        segment_b={"pickup_zone_ids": 138},
        metric="fare_revenue",
        name_a="JFK Airport",
        name_b="LaGuardia Airport",
    )

    print(json.dumps(chart, indent=2, default=str))

    print("\n5. VALIDATION TEST")

    try:
        charts.build_chart({
            "metric": "unknown_metric",
            "group_by": "hour",
            "chart_type": "bar",
        })
    except ValueError as error:
        print("Correctly rejected invalid request:", error)

    print("\n" + "=" * 65)
    print("ALL CHART SERVICE TESTS COMPLETED")
    print("=" * 65)

finally:
    analytics.close()
