from src.analytics.engine import MoveIQAnalytics

analytics = MoveIQAnalytics()

try:
    print("\n1. MANHATTAN VS QUEENS")
    print(
        analytics.compare_segments(
            {"borough": "Manhattan"},
            {"borough": "Queens"},
            "Manhattan",
            "Queens",
        ).to_string(index=False)
    )

    print("\n2. WEEKDAY VS WEEKEND")
    print(
        analytics.compare_segments(
            {"weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]},
            {"weekdays": ["Saturday", "Sunday"]},
            "Weekdays",
            "Weekends",
        ).to_string(index=False)
    )

    print("\n3. JFK VS LAGUARDIA")
    print(
        analytics.compare_segments(
            {"pickup_zone_ids": 132},
            {"pickup_zone_ids": 138},
            "JFK Airport",
            "LaGuardia Airport",
        ).to_string(index=False)
    )

    print("\n4. MORNING VS EVENING")
    print(
        analytics.compare_segments(
            {"hours": [6, 7, 8, 9, 10, 11]},
            {"hours": [17, 18, 19, 20, 21, 22]},
            "Morning",
            "Evening",
        ).to_string(index=False)
    )

    print("\n5. DROPOFF ZONE GROUPING CHECK")
    print(
        analytics.get_filtered_demand(
            group_by="dropoff_zone",
            borough="Manhattan",
        ).head(10).to_string(index=False)
    )

    print("\nALL COMPARISON TESTS COMPLETED")

finally:
    analytics.close()
