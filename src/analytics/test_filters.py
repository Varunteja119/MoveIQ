from src.analytics.engine import MoveIQAnalytics

analytics = MoveIQAnalytics()

try:
    print("\n" + "=" * 65)
    print("MOVEIQ — DYNAMIC FILTER TESTS")
    print("=" * 65)

    print("\n1. HOURLY DEMAND IN MANHATTAN")
    print(
        analytics.get_filtered_demand(
            group_by="hour",
            borough="Manhattan"
        ).to_string(index=False)
    )

    print("\n2. FRIDAY REVENUE")
    print(
        analytics.get_filtered_revenue(
            group_by="hour",
            weekdays="Friday"
        ).to_string(index=False)
    )

    print("\n3. EVENING DEMAND: 5 PM TO 9 PM")
    print(
        analytics.get_filtered_demand(
            group_by="hour",
            hours=[17, 18, 19, 20, 21]
        ).to_string(index=False)
    )

    print("\n4. REVENUE BY BOROUGH")
    print(
        analytics.get_filtered_revenue(
            group_by="borough"
        ).to_string(index=False)
    )

    print("\n5. JFK AIRPORT REVENUE")
    print(
        analytics.get_filtered_revenue(
            group_by="hour",
            pickup_zone_ids=132
        ).to_string(index=False)
    )

    print("\n6. WEEKEND DEMAND IN QUEENS")
    print(
        analytics.get_filtered_demand(
            group_by="hour",
            weekdays=["Saturday", "Sunday"],
            borough="Queens"
        ).to_string(index=False)
    )

    print("\n7. DATE RANGE FILTER")
    print(
        analytics.get_filtered_demand(
            group_by="weekday",
            start_date="2025-01-01",
            end_date="2025-01-07"
        ).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("DYNAMIC FILTER TESTS COMPLETED")
    print("=" * 65)

finally:
    analytics.close()
