"""MoveIQ FastAPI application with isolated Redis-backed Copilot sessions."""

import json
import math
import uuid
from datetime import date
from numbers import Real
from contextlib import asynccontextmanager
from threading import Lock, RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.ai.agent import MoveIQAgent
from src.analytics.historical_engine import HistoricalAnalytics


SESSION_TTL_SECONDS = 86400
SESSION_KEY_PREFIX = "moveiq:conversation:"


class CopilotRequest(BaseModel):
    session_id: str = Field(min_length=36, max_length=36)
    question: str = Field(min_length=3, max_length=4000)


class SegmentComparisonRequest(BaseModel):
    segment_a: dict[str, Any]
    segment_b: dict[str, Any]
    name_a: str = "Segment A"
    name_b: str = "Segment B"


class ZoneForecastRequest(BaseModel):
    zone_ids: list[int] = Field(min_length=1, max_length=263)
    as_of: str | None = None


class ForecastInsightsRequest(BaseModel):
    scope: str = "citywide"
    borough: str | None = None
    zone_ids: list[int] | None = None
    as_of: str | None = None


class HistoricalComparisonRequest(BaseModel):
    location_id: int = Field(ge=1, le=263)
    segment_a_start: date
    segment_a_end: date
    segment_b_start: date
    segment_b_end: date
    location_role: str = "pickup"
    hour: int | None = Field(default=None, ge=0, le=23)


def create_app(agent_factory=MoveIQAgent, historical_factory=HistoricalAnalytics):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent = agent_factory()
        historical = historical_factory()
        app.state.agent = agent
        app.state.historical_analytics = historical
        app.state.historical_lock = RLock()
        app.state.service_lock = RLock()
        app.state.toolbox_lock = RLock()
        app.state.session_locks = {}
        app.state.session_locks_guard = Lock()

        agent.tool_lock = app.state.toolbox_lock

        try:
            yield
        finally:
            try:
                historical.close()
            finally:
                agent.close()

    app = FastAPI(
        title="MoveIQ API",
        description=(
            "Mobility analytics, demand forecasting "
            "and session-based AI Copilot."
        ),
        version="1.1.0",
        lifespan=lifespan,
    )

    def get_agent():
        agent = getattr(app.state, "agent", None)

        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="MoveIQ services are not ready.",
            )

        return agent

    def get_session_key(session_id: str):
        try:
            # Validate UUID format before constructing the Redis key.
            normalized_id = str(uuid.UUID(session_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID.",
            ) from exc

        return f"{SESSION_KEY_PREFIX}{normalized_id}"

    def get_redis(agent):
        return agent.context_budget.redis

    def get_session_lock(session_id):
        normalized_id = str(uuid.UUID(session_id))

        with app.state.session_locks_guard:
            lock = app.state.session_locks.get(normalized_id)

            if lock is None:
                lock = Lock()
                app.state.session_locks[normalized_id] = lock

        return lock

    def run_service(operation):
        agent = get_agent()

        try:
            with app.state.toolbox_lock:
                return operation(agent)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The requested operation failed. "
                    "Check the API server logs."
                ),
            ) from exc
    def get_historical():
        service = getattr(app.state, "historical_analytics", None)

        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Historical analytics are not ready.",
            )

        return service

    def run_historical_service(operation):
        service = get_historical()

        try:
            with app.state.historical_lock:
                return operation(service)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The historical analytics operation failed. "
                    "Check the API server logs."
                ),
            ) from exc

    # ---------------------------------------------------------
    # SYSTEM
    # ---------------------------------------------------------

    @app.get("/health", tags=["System"])
    def health():
        ready = getattr(app.state, "agent", None) is not None

        return {
            "status": "ok" if ready else "starting",
            "ready": ready,
        }

    # ---------------------------------------------------------
    # ANALYTICS
    # ---------------------------------------------------------

    @app.get("/api/v1/analytics/overview", tags=["Analytics"])
    def analytics_overview():
        return run_service(
            lambda agent: agent.toolbox.get_dataset_overview()
        )

    @app.get("/api/v1/analytics/trip-metrics", tags=["Analytics"])
    def trip_metrics():
        return run_service(
            lambda agent: agent.toolbox.get_trip_metrics()
        )

    @app.get("/api/v1/analytics/demand", tags=["Analytics"])
    def demand_analysis(
        group_by: str = "hour",
        start_date: str | None = None,
        end_date: str | None = None,
        weekdays: list[int] | None = Query(default=None),
        hours: list[int] | None = Query(default=None),
        borough: str | None = None,
        pickup_zone_ids: list[int] | None = Query(default=None),
        dropoff_zone_ids: list[int] | None = Query(default=None),
    ):
        return run_service(
            lambda agent: agent.toolbox.get_demand_analysis(
                group_by=group_by,
                start_date=start_date,
                end_date=end_date,
                weekdays=weekdays,
                hours=hours,
                borough=borough,
                pickup_zone_ids=pickup_zone_ids,
                dropoff_zone_ids=dropoff_zone_ids,
            )
        )

    @app.get("/api/v1/analytics/revenue", tags=["Analytics"])
    def revenue_analysis(
        group_by: str = "borough",
        start_date: str | None = None,
        end_date: str | None = None,
        weekdays: list[int] | None = Query(default=None),
        hours: list[int] | None = Query(default=None),
        borough: str | None = None,
        pickup_zone_ids: list[int] | None = Query(default=None),
        dropoff_zone_ids: list[int] | None = Query(default=None),
    ):
        return run_service(
            lambda agent: agent.toolbox.get_revenue_analysis(
                group_by=group_by,
                start_date=start_date,
                end_date=end_date,
                weekdays=weekdays,
                hours=hours,
                borough=borough,
                pickup_zone_ids=pickup_zone_ids,
                dropoff_zone_ids=dropoff_zone_ids,
            )
        )

    @app.get("/api/v1/analytics/zones", tags=["Analytics"])
    def zone_rankings(
        kind: str = "pickup",
        limit: int = Query(default=10, ge=1, le=300),
    ):
        return run_service(
            lambda agent: agent.toolbox.get_zone_rankings(
                kind=kind,
                limit=limit,
            )
        )

    @app.post("/api/v1/analytics/compare", tags=["Analytics"])
    def compare_segments(request: SegmentComparisonRequest):
        return run_service(
            lambda agent: agent.toolbox.compare_segments(
                segment_a=request.segment_a,
                segment_b=request.segment_b,
                name_a=request.name_a,
                name_b=request.name_b,
            )
        )

    # ---------------------------------------------------------
    # FORECASTS
    # ---------------------------------------------------------

    @app.get("/api/v1/forecasts/citywide", tags=["Forecasts"])
    def citywide_forecast(as_of: str | None = None):
        return run_service(
            lambda agent: agent.toolbox.get_citywide_forecast(
                as_of=as_of
            )
        )

    @app.post("/api/v1/forecasts/zones", tags=["Forecasts"])
    def zone_forecast(request: ZoneForecastRequest):
        return run_service(
            lambda agent: agent.toolbox.get_zone_forecast(
                zone_ids=request.zone_ids,
                as_of=request.as_of,
            )
        )

    @app.get(
        "/api/v1/forecasts/boroughs/{borough}",
        tags=["Forecasts"],
    )
    def borough_forecast(
        borough: str,
        as_of: str | None = None,
    ):
        return run_service(
            lambda agent: agent.toolbox.get_borough_forecast(
                borough=borough,
                as_of=as_of,
            )
        )

    @app.post("/api/v1/forecasts/insights", tags=["Forecasts"])
    def forecast_insights(request: ForecastInsightsRequest):
        return run_service(
            lambda agent: agent.toolbox.get_forecast_insights(
                scope=request.scope,
                borough=request.borough,
                zone_ids=request.zone_ids,
                as_of=request.as_of,
            )
        )

    # ---------------------------------------------------------
    # COPILOT SESSION MANAGEMENT
    # ---------------------------------------------------------

    @app.post("/api/v1/copilot/sessions", tags=["Copilot"])
    def create_session():
        def create(agent):
            redis = get_redis(agent)
            session_id = str(uuid.uuid4())
            session_key = get_session_key(session_id)

            # Start with the agent's normal system prompt.
            agent.reset()

            redis.setex(
                session_key,
                SESSION_TTL_SECONDS,
                json.dumps(agent.messages, ensure_ascii=False),
            )

            return {
                "session_id": session_id,
                "expires_in_seconds": SESSION_TTL_SECONDS,
                "message": "Copilot session created.",
            }

        agent = get_agent()
        try:
            with app.state.service_lock:
                return create(agent)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Could not create the Copilot session.",
            ) from exc

    @app.delete(
        "/api/v1/copilot/sessions/{session_id}",
        tags=["Copilot"],
    )
    def delete_session(session_id: str):
        agent = get_agent()
        session_key = get_session_key(session_id)

        with get_session_lock(session_id):
            deleted = get_redis(agent).delete(session_key)

            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail="Session not found or already expired.",
                )

        return {"message": "Copilot session deleted."}

    @app.post(
        "/api/v1/copilot/sessions/{session_id}/reset",
        tags=["Copilot"],
    )
    def reset_session(session_id: str):
        agent = get_agent()
        redis = get_redis(agent)
        session_key = get_session_key(session_id)

        with get_session_lock(session_id):
            saved_history = redis.get(session_key)

            if saved_history is None:
                raise HTTPException(
                    status_code=404,
                    detail="Session not found or expired.",
                )

            if isinstance(saved_history, bytes):
                saved_history = saved_history.decode("utf-8")

            history = json.loads(saved_history)

            if not history or history[0].get("role") != "system":
                raise HTTPException(
                    status_code=500,
                    detail="Stored conversation history is invalid.",
                )

            redis.setex(
                session_key,
                SESSION_TTL_SECONDS,
                json.dumps([history[0]], ensure_ascii=False),
            )

        return {"message": "Conversation history reset."}

    # ---------------------------------------------------------
    # COPILOT CHAT
    # ---------------------------------------------------------

    @app.post("/api/v1/copilot/ask", tags=["Copilot"])
    def copilot_ask(request: CopilotRequest):
        base_agent = get_agent()
        redis = get_redis(base_agent)
        session_key = get_session_key(request.session_id)

        with get_session_lock(request.session_id):
            saved_history = redis.get(session_key)

            if saved_history is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Session not found or expired. "
                        "Create a new session."
                    ),
                )

            if isinstance(saved_history, bytes):
                saved_history = saved_history.decode("utf-8")

            try:
                history = json.loads(saved_history)

                if (
                    not isinstance(history, list)
                    or not history
                    or history[0].get("role") != "system"
                ):
                    raise ValueError("Stored conversation history is invalid.")

                spawn = getattr(base_agent, "spawn", None)

                if callable(spawn):
                    request_agent = spawn()
                    owns_request_agent = True
                else:
                    request_agent = base_agent
                    owns_request_agent = False

                try:
                    request_agent.messages = history
                    answer = request_agent.ask(request.question)

                    redis.setex(
                        session_key,
                        SESSION_TTL_SECONDS,
                        json.dumps(
                            request_agent.messages,
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    )

                    return {
                        "session_id": request.session_id,
                        "answer": answer,
                        "expires_in_seconds": SESSION_TTL_SECONDS,
                    }

                finally:
                    if owns_request_agent:
                        request_agent.close()
                    else:
                        request_agent.reset()

            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "The Copilot request failed. "
                        "Check the API server logs."
                    ),
                ) from exc

    # ---------------------------------------------------------
    # HISTORICAL ANALYTICS
    # ---------------------------------------------------------

    @app.get("/api/v1/historical/overview", tags=["Historical Analytics"])
    def historical_overview():
        return run_historical_service(
            lambda service: service.get_overview()
        )


    @app.get("/api/v1/historical/patterns", tags=["Historical Analytics"])
    def historical_patterns():
        return run_historical_service(
            lambda service: service.get_patterns_summary()
        )

    @app.get("/api/v1/historical/zones/{location_id}", tags=["Historical Analytics"])
    def historical_zone_profile(
        location_id: int,
        location_role: str = Query(default="pickup", pattern="^(pickup|dropoff)$"),
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        return run_historical_service(
            lambda service: service.get_zone_profile(
                location_id=location_id,
                location_role=location_role,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
        )

    @app.get("/api/v1/historical/zones/{location_id}/hourly", tags=["Historical Analytics"])
    def historical_zone_hourly(
        location_id: int,
        location_role: str = Query(default="pickup", pattern="^(pickup|dropoff)$"),
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        records = run_historical_service(
            lambda service: service.get_zone_hourly(
                location_id=location_id,
                location_role=location_role,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
        )
        for record in records:
            for key, value in record.items():
                if isinstance(value, Real) and not math.isfinite(value):
                    record[key] = None

        return {
            "location_id": location_id,
            "location_role": location_role,
            "time_basis": "pickup_hour",
            "records": records,
        }

    @app.post("/api/v1/historical/compare", tags=["Historical Analytics"])
    def historical_compare(request: HistoricalComparisonRequest):
        if request.segment_a_start > request.segment_a_end:
            raise HTTPException(400, "Segment A start must not be after its end.")
        if request.segment_b_start > request.segment_b_end:
            raise HTTPException(400, "Segment B start must not be after its end.")
        if request.location_role not in {"pickup", "dropoff"}:
            raise HTTPException(400, "location_role must be pickup or dropoff.")
        if not (
            request.segment_a_end < request.segment_b_start
            or request.segment_b_end < request.segment_a_start
        ):
            raise HTTPException(400, "Comparison date ranges must not overlap.")

        return run_historical_service(
            lambda service: service.compare_periods(
                location_id=request.location_id,
                segment_a_start=request.segment_a_start.isoformat(),
                segment_a_end=request.segment_a_end.isoformat(),
                segment_b_start=request.segment_b_start.isoformat(),
                segment_b_end=request.segment_b_end.isoformat(),
                location_role=request.location_role,
                hour=request.hour,
            )
        )

    return app


app = create_app()
