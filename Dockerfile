FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# XGBoost requires the OpenMP runtime library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src/ ./src/

COPY data/processed/yellow_tripdata_2025-01_cleaned.parquet ./data/processed/
COPY data/reference/taxi_zone_lookup.csv ./data/reference/

COPY data/analytics/historical/zone_hourly_2024_2026.parquet ./data/analytics/historical/

COPY data/ml/processed/hourly_demand_features_2024_2026.parquet ./data/ml/processed/
COPY data/ml/processed/zone_hourly_demand_features_2024_2026.parquet ./data/ml/processed/

COPY models/moveiq_direct_demand_forecaster_2024_2026_candidate.joblib ./models/
COPY models/moveiq_zone_direct_demand_forecaster_2024_2026_candidate.joblib ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
