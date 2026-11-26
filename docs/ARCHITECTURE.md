# MoveIQ - System Architecture

## 1. Overview

MoveIQ is an AI-powered mobility and operations intelligence platform built using NYC Yellow Taxi trip data.

The system combines data engineering, analytical processing, machine learning, geospatial visualization, backend APIs, and a conversational AI interface.

## 2. High-Level Architecture

```text
                  NYC TLC Dataset
                         |
                         v
              Data Processing Pipeline
                         |
                 Cleaning & Validation
                         |
                         v
                  Parquet Storage
                         |
                         v
                  DuckDB Engine
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Analytics     Forecasting    Historical
       Services       Services      Insights
          |              |              |
          +--------------+--------------+
                         |
                         v
                    FastAPI
                         |
             +-----------+-----------+
             |                       |
             v                       v
        React Frontend          AI Copilot
             |                       |
      +------+------+             LangGraph
      |      |      |                 |
      v      v      v              Groq API
   Analytics Map Forecast             |
                                      v
                                    Redis
                              Session Management

```
