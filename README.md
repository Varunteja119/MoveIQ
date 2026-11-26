# MoveIQ
### AI-Powered Mobility & Operations Intelligence

MoveIQ is an end-to-end mobility analytics platform built using NYC taxi trip data. It combines data engineering, business intelligence, machine learning, geospatial analytics, and generative AI.

## Core Features

- **Mobility Analytics:** Trip volume, fares, distances, duration, revenue, and demand patterns.
- **Historical Insights:** Hourly zone analysis and historical period comparisons.
- **Demand Forecasting:** Citywide and taxi-zone hourly demand forecasts.
- **Interactive NYC Map:** Geographic visualization of pickup demand across taxi zones.
- **AI Analytics Copilot:** Natural-language questions, analytical tools, and conversational sessions.
- **Backend API:** FastAPI endpoints for analytics, forecasts, historical insights, and Copilot functionality.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.12 |
| Data processing | Pandas, NumPy, PyArrow |
| Analytics | DuckDB, SQL |
| Storage | Parquet |
| Machine learning | Scikit-learn, XGBoost |
| Backend | FastAPI |
| AI | LangGraph, Groq API |
| Frontend | React, TypeScript, Vite |
| Mapping | MapLibre GL |
| Session infrastructure | Redis |
| Deployment | Docker, Docker Compose |
| Testing | Pytest, ESLint |

## Data Sources

1. NYC Taxi & Limousine Commission January 2025 Yellow Taxi trip records.
2. UCI Bike Sharing Dataset.

The primary analytics implementation uses the NYC TLC dataset.

## Architecture

```text
NYC TLC Parquet Data
        |
        v
Data Processing & Validation
        |
        v
DuckDB Analytics Engine
        |
        +--------------------+
        |                    |
        v                    v
Analytics & Historical   Forecasting
Insights                 Services
        |                    |
        +---------+----------+
                  |
                  v
               FastAPI
                  |
          +-------+-------+
          |               |
          v               v
    React Frontend    AI Copilot
                          |
                       LangGraph
                          |
                       Groq API
                          |
                        Redis
```
## License

To be determined.

## Docker data prerequisites

The Docker image requires locally generated NYC TLC Parquet datasets and trained forecasting models. These large artifacts are intentionally excluded from Git. Before running Docker Compose, download and prepare the historical data, generate the analytics and ML feature datasets, and train the forecasting models. Place the resulting files at the exact paths referenced by the Dockerfile. See the scripts under src/data and src/ml for the preparation and training workflows. The Docker image cannot be built from a fresh clone until these artifacts are generated.

