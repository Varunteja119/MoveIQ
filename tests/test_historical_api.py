from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeRedis:
    def get(self, key):
        return None

    def setex(self, key, ttl, value):
        return True

    def delete(self, key):
        return 0

    def exists(self, key):
        return 0


class FakeContextBudget:
    def __init__(self):
        self.redis = FakeRedis()


class FakeToolbox:
    pass


class FakeAgent:
    def __init__(self):
        self.toolbox = FakeToolbox()
        self.context_budget = FakeContextBudget()
        self.tool_lock = None

    def close(self):
        pass


def test_historical_overview():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.get("/api/v1/historical/overview")

    assert response.status_code == 200
    data = response.json()

    assert data["zone_count"] == 263
    assert data["total_trips"] == 123471633


def test_historical_zone_profile():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.get("/api/v1/historical/zones/161")

    assert response.status_code == 200
    data = response.json()

    assert data["zone_name"] == "Midtown Center"
    assert data["borough"] == "Manhattan"
    assert data["total_trips"] > 0
    assert data["average_fare"] > 0
    assert data["average_duration_minutes"] > 0


def test_historical_zone_profile_date_filter():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.get(
            "/api/v1/historical/zones/161",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-03",
            },
        )

    assert response.status_code == 200
    data = response.json()

    assert data["first_date"].startswith("2025-01-")
    assert data["last_date"].startswith("2025-01-")


def test_historical_hourly_records():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.get(
            "/api/v1/historical/zones/161/hourly",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-01",
            },
        )

    assert response.status_code == 200
    data = response.json()

    assert data["location_role"] == "pickup"
    assert data["time_basis"] == "pickup_hour"
    assert len(data["records"]) > 0
    assert data["records"][0]["pickup_hour"].startswith("2025-01-01")


def test_historical_period_comparison():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.post(
            "/api/v1/historical/compare",
            json={
                "location_id": 161,
                "segment_a_start": "2025-01-01",
                "segment_a_end": "2025-01-03",
                "segment_b_start": "2025-01-04",
                "segment_b_end": "2025-01-06",
                "location_role": "pickup",
            },
        )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 2
    assert {item["segment"] for item in data} == {
        "segment_a",
        "segment_b",
    }


def test_historical_comparison_rejects_overlapping_periods():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.post(
            "/api/v1/historical/compare",
            json={
                "location_id": 161,
                "segment_a_start": "2025-01-01",
                "segment_a_end": "2025-01-05",
                "segment_b_start": "2025-01-04",
                "segment_b_end": "2025-01-08",
            },
        )

    assert response.status_code == 400
