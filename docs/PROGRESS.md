# MoveIQ
### AI-Powered Mobility & Operations Intelligence

MoveIQ is a mobility analytics and intelligence platform combining
business intelligence, predictive machine learning, generative AI,
and backend engineering.

## Objectives

- Analyze mobility demand and operational trends.
- Build SQL-powered business intelligence.
- Forecast demand and predict trip duration.
- Provide an AI-powered analytics copilot.
- Support safe natural-language-to-SQL interactions.
- Retrieve information from operational documentation.

## Technology Stack

- Python, Pandas, NumPy
- PostgreSQL, SQLAlchemy
- Power BI, DAX
- Scikit-learn, XGBoost
- FastAPI
- LangGraph, Groq API
- Streamlit, Plotly
- Docker, Pytest, GitHub Actions

## Data Sources

1. NYC Taxi & Limousine Commission trip records.
2. UCI Bike Sharing Dataset.

## Project Structure

- data/: Raw and processed datasets.
- notebooks/: Exploratory analysis.
- src/: Application source code.
- tests/: Automated tests.
- docs/: Technical documentation.
- dashboard/: Streamlit dashboard.
- models/: Trained ML artifacts.
- scripts/: Reusable scripts.

## Development

Python 3.12 is used in an isolated virtual environment.

Activate it on Windows:

`powershell
.\.venv\Scripts\Activate.ps1

### 2.4 Create our progress log

We'll use this to preserve continuity between development sessions.

`powershell
@"
# MoveIQ Development Progress

## Project configuration
- Project: MoveIQ
- Development OS: Windows
- Terminal: PowerShell
- Python: 3.12.5
- Git: Initialized
- Virtual environment: .venv
- Docker: Installed

## Completed
- [x] Project objectives defined
- [x] Technology stack selected
- [x] Python environment configured
- [x] Initial dependencies installed
- [x] Core imports verified
- [x] Git repository initialized

## In progress
- [ ] Repository documentation
- [ ] Architecture documentation
- [ ] Dataset acquisition
- [ ] Data inspection and profiling

## Memory-efficiency decisions
- Process limited dataset samples initially.
- Prefer Parquet for analytical storage.
- Select only required columns.
- Use chunked processing where appropriate.
- Avoid unnecessary local LLM hosting.
- Run only necessary Docker services.

## Current milestone
Week 1 - Day 1: Project foundation

## Next milestone
Week 1 - Day 2: NYC TLC dataset acquisition and inspection.

## Experimental results
None yet. No ML experiments have been conducted.
