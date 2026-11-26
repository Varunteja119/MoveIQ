from src.ml.forecast_service import MoveIQForecastService


def main():
    service = MoveIQForecastService()
    forecast = service.forecast_next_24_hours()

    assert len(forecast) == 24, "Expected exactly 24 hourly forecasts."
    assert forecast["predicted_pickups"].notna().all()
    assert (forecast["predicted_pickups"] >= 0).all()
    assert forecast["forecast_timestamp"].is_monotonic_increasing
    assert forecast["forecast_timestamp"].is_unique

    expected_first = (
        service.history.iloc[-1]["pickup_hour"]
        + __import__("pandas").Timedelta(hours=1)
    )

    assert forecast.iloc[0]["forecast_timestamp"] == expected_first

    print("PASS: Exactly 24 hourly forecasts generated.")
    print("PASS: Forecast timestamps are ordered and unique.")
    print("PASS: Predictions contain no missing or negative values.")
    print("\nFirst 5 predictions:")
    print(forecast.head().to_string(index=False))
    print("\nLast 5 predictions:")
    print(forecast.tail().to_string(index=False))


if __name__ == "__main__":
    main()
