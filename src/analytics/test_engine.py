from src.analytics.engine import MoveIQAnalytics


def main():

    engine = MoveIQAnalytics()

    try:
        print("\n" + "=" * 60)
        print("MOVEIQ ANALYTICS ENGINE — TEST RUN")
        print("=" * 60)

        print("\n1. DATASET OVERVIEW")
        print(engine.get_overview().to_string(index=False))

        print("\n2. HOURLY DEMAND")
        print(engine.get_hourly_demand().to_string(index=False))

        print("\n3. WEEKDAY DEMAND")
        print(engine.get_weekday_demand().to_string(index=False))

        print("\n4. TOP 5 PICKUP NEIGHBORHOODS")
        print(
            engine.get_top_pickup_zones(5)
            .to_string(index=False)
        )

        print("\n5. TOP 5 DROPOFF NEIGHBORHOODS")
        print(
            engine.get_top_dropoff_zones(5)
            .to_string(index=False)
        )

        print("\n6. TRIP METRICS")
        print(engine.get_trip_metrics().to_string(index=False))

        print("\n7. PAYMENT DISTRIBUTION")
        print(
            engine.get_payment_distribution()
            .to_string(index=False)
        )

        print("\n8. NEIGHBORHOOD-HOUR DEMAND")
        print(
            engine.get_zone_hour_demand(5)
            .head(15)
            .to_string(index=False)
        )

        print("\n" + "=" * 60)
        print("ALL ANALYTICS ENGINE TESTS COMPLETED")
        print("=" * 60)

    finally:
        engine.close()


if __name__ == "__main__":
    main()
