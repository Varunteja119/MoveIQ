import json
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value
        return True

    def delete(self, key):
        return int(self.data.pop(key, None) is not None)

    def exists(self, key):
        return int(key in self.data)


class FakeToolbox:
    def get_dataset_overview(self):
        return {
            "tool": "get_dataset_overview",
            "data": [{"total_trips": 100}],
        }

    def get_trip_metrics(self):
        return {
            "tool": "get_trip_metrics",
            "data": [{"average_fare": 12.5}],
        }


class FakeContextBudget:
    def __init__(self):
        self.redis = FakeRedis()


class FakeAgent:
    def __init__(self):
        self.toolbox = FakeToolbox()
        self.context_budget = FakeContextBudget()
        self.messages = []
        self.closed = False
        self.reset()

    def reset(self):
        self.messages = [
            {"role": "system", "content": "MoveIQ system prompt"}
        ]

    def ask(self, question):
        self.messages.append({
            "role": "user",
            "content": question,
        })

        answer = f"Answered: {question}"

        self.messages.append({
            "role": "assistant",
            "content": answer,
        })

        return answer

    def close(self):
        self.closed = True



class ConcurrentRequestAgent:
    """Request-local fake agent sharing a barrier across sessions."""

    def __init__(self, toolbox, context_budget, barrier):
        self.toolbox = toolbox
        self.context_budget = context_budget
        self.barrier = barrier
        self.messages = []

    def ask(self, question):
        self.messages.append({
            "role": "user",
            "content": question,
        })

        # Both requests must arrive here before either can finish.
        self.barrier.wait(timeout=5)

        answer = f"Answered: {question}"

        self.messages.append({
            "role": "assistant",
            "content": answer,
        })

        return answer

    def close(self):
        pass


class ConcurrentFakeAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.barrier = threading.Barrier(2)

    def spawn(self):
        return ConcurrentRequestAgent(
            toolbox=self.toolbox,
            context_budget=self.context_budget,
            barrier=self.barrier,
        )

def test_health_endpoint():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["ready"] is True


def test_analytics_overview():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.get("/api/v1/analytics/overview")

        assert response.status_code == 200
        assert response.json()["data"][0]["total_trips"] == 100


def test_create_session():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.post("/api/v1/copilot/sessions")

        assert response.status_code == 200

        body = response.json()

        assert body["session_id"]
        assert body["expires_in_seconds"] == 86400


def test_sessions_keep_separate_conversation_histories():
    with TestClient(create_app(FakeAgent)) as client:
        session_a = client.post(
            "/api/v1/copilot/sessions"
        ).json()["session_id"]

        session_b = client.post(
            "/api/v1/copilot/sessions"
        ).json()["session_id"]

        first = client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": session_a,
                "question": "Show me pickup demand",
            },
        )

        second = client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": session_b,
                "question": "Show me revenue trends",
            },
        )

        assert first.status_code == 200
        assert second.status_code == 200

        agent = client.app.state.agent
        redis = agent.context_budget.redis

        history_a = json.loads(
            redis.get(f"moveiq:conversation:{session_a}")
        )

        history_b = json.loads(
            redis.get(f"moveiq:conversation:{session_b}")
        )

        assert any(
            message.get("content") == "Show me pickup demand"
            for message in history_a
        )

        assert not any(
            message.get("content") == "Show me revenue trends"
            for message in history_a
        )

        assert any(
            message.get("content") == "Show me revenue trends"
            for message in history_b
        )

        assert not any(
            message.get("content") == "Show me pickup demand"
            for message in history_b
        )


def test_follow_up_uses_same_session_history():
    with TestClient(create_app(FakeAgent)) as client:
        session_id = client.post(
            "/api/v1/copilot/sessions"
        ).json()["session_id"]

        client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": session_id,
                "question": "Which zone has the highest demand?",
            },
        )

        client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": session_id,
                "question": "What about its evening demand?",
            },
        )

        history = json.loads(
            client.app.state.agent.context_budget.redis.get(
                f"moveiq:conversation:{session_id}"
            )
        )

        user_messages = [
            message["content"]
            for message in history
            if message.get("role") == "user"
        ]

        assert user_messages == [
            "Which zone has the highest demand?",
            "What about its evening demand?",
        ]


def test_unknown_session_is_rejected():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": "00000000-0000-0000-0000-000000000000",
                "question": "Show me demand",
            },
        )

        assert response.status_code == 404


def test_reset_session():
    with TestClient(create_app(FakeAgent)) as client:
        session_id = client.post(
            "/api/v1/copilot/sessions"
        ).json()["session_id"]

        client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": session_id,
                "question": "Show me demand",
            },
        )

        response = client.post(
            f"/api/v1/copilot/sessions/{session_id}/reset"
        )

        assert response.status_code == 200

        history = json.loads(
            client.app.state.agent.context_budget.redis.get(
                f"moveiq:conversation:{session_id}"
            )
        )

        assert len(history) == 1
        assert history[0]["role"] == "system"


def test_delete_session():
    with TestClient(create_app(FakeAgent)) as client:
        session_id = client.post(
            "/api/v1/copilot/sessions"
        ).json()["session_id"]

        response = client.delete(
            f"/api/v1/copilot/sessions/{session_id}"
        )

        assert response.status_code == 200

        follow_up = client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": session_id,
                "question": "Show me demand",
            },
        )

        assert follow_up.status_code == 404


def test_invalid_question_is_rejected():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.post(
            "/api/v1/copilot/ask",
            json={
                "session_id": "00000000-0000-0000-0000-000000000000",
                "question": "x",
            },
        )

        assert response.status_code == 422


def test_different_sessions_can_process_requests_concurrently():
    with TestClient(create_app(ConcurrentFakeAgent)) as client:
        session_a = client.post(
            "/api/v1/copilot/sessions"
        ).json()["session_id"]

        session_b = client.post(
            "/api/v1/copilot/sessions"
        ).json()["session_id"]

        def ask(session_id, question):
            return client.post(
                "/api/v1/copilot/ask",
                json={
                    "session_id": session_id,
                    "question": question,
                },
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                ask, session_a, "Show me pickup demand"
            )
            future_b = executor.submit(
                ask, session_b, "Show me revenue trends"
            )

            response_a = future_a.result(timeout=10)
            response_b = future_b.result(timeout=10)

        assert response_a.status_code == 200
        assert response_b.status_code == 200

        assert response_a.json()["answer"] == (
            "Answered: Show me pickup demand"
        )
        assert response_b.json()["answer"] == (
            "Answered: Show me revenue trends"
        )


def test_historical_analytics_endpoints():
    with TestClient(create_app(FakeAgent)) as client:
        overview = client.get("/api/v1/historical/overview")
        assert overview.status_code == 200
        data = overview.json()
        assert data["first_date"].startswith("2024-01-01")
        assert data["last_date"].startswith("2026-09-30")
        assert data["zone_count"] == 263
        assert data["total_trips"] == 123471633.0

        profile = client.get(
            "/api/v1/historical/zones/161",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-01",
            },
        )
        assert profile.status_code == 200
        data = profile.json()
        assert data["zone_name"] == "Midtown Center"
        assert data["total_trips"] > 0
        assert data["average_fare"] > 0
        assert data["average_distance_miles"] > 0
        assert data["average_duration_minutes"] > 0

        hourly = client.get(
            "/api/v1/historical/zones/161/hourly",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-01",
            },
        )
        assert hourly.status_code == 200
        data = hourly.json()
        assert data["time_basis"] == "pickup_hour"
        assert data["location_role"] == "pickup"
        assert data["records"]
        assert all(
            0 <= row["pickup_hour_of_day"] <= 23
            for row in data["records"]
        )

        dropoff = client.get(
            "/api/v1/historical/zones/161",
            params={
                "location_role": "dropoff",
                "start_date": "2025-01-01",
                "end_date": "2025-01-01",
            },
        )
        assert dropoff.status_code == 200
        assert dropoff.json()["total_trips"] > 0

        comparison = client.post(
            "/api/v1/historical/compare",
            json={
                "location_id": 161,
                "segment_a_start": "2025-01-01",
                "segment_a_end": "2025-01-01",
                "segment_b_start": "2025-01-02",
                "segment_b_end": "2025-01-02",
            },
        )
        assert comparison.status_code == 200
        data = comparison.json()
        assert {row["segment"] for row in data} == {
            "segment_a", "segment_b"
        }
        assert all(row["total_trips"] > 0 for row in data)


def test_historical_comparison_rejects_overlapping_periods():
    with TestClient(create_app(FakeAgent)) as client:
        response = client.post(
            "/api/v1/historical/compare",
            json={
                "location_id": 161,
                "segment_a_start": "2025-01-01",
                "segment_a_end": "2025-01-05",
                "segment_b_start": "2025-01-05",
                "segment_b_end": "2025-01-10",
            },
        )
        assert response.status_code == 400
