from src.analytics.engine import MoveIQAnalytics

analytics = MoveIQAnalytics()

try:
    print("\n" + "=" * 60)
    print("MOVEIQ — KPI TEST")
    print("=" * 60)

    print("\n1. OVERALL REVENUE METRICS")
    print(analytics.get_revenue_metrics().to_string(index=False))

    print("\n2. HOURLY REVENUE")
    print(analytics.get_hourly_revenue().to_string(index=False))

    print("\n3. WEEKDAY REVENUE")
    print(analytics.get_weekday_revenue().to_string(index=False))

    print("\n4. TOP 10 NEIGHBORHOODS BY FARE REVENUE")
    print(analytics.get_zone_revenue(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("ALL KPI TESTS COMPLETED")
    print("=" * 60)

finally:
    analytics.close()
