FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src/ ./src/

COPY data/reference/taxi_zone_lookup.csv ./data/reference/

RUN curl -fL \
    "https://github.com/Varunteja119/MoveIQ/releases/download/v1.0.0/moveiq-deployment-data.zip" \
    -o /tmp/moveiq-deployment-data.zip \
    && unzip -q /tmp/moveiq-deployment-data.zip -d /app \
    && rm /tmp/moveiq-deployment-data.zip

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]