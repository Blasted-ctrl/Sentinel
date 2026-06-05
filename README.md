<div align="center">

# Sentinel

### Wildfire risk forecasting engine

**Ingests live NOAA/NASA weather and satellite data, trains a CNN on satellite imagery and an
LSTM on climate time-series to predict wildfire ignition risk by region, and serves real-time
risk scores through a live map dashboard.**

[![CI](https://github.com/Blasted-ctrl/Sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/Blasted-ctrl/Sentinel/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-orange)

</div>

---

## Problem

Wildfires kill people and destroy communities, and the window between ignition and uncontrollable
spread is short. Emergency planners need **earlier, regional risk signals** so they can pre-position
crews and issue evacuation orders sooner. Sentinel turns public earth-observation data into a daily,
per-region ignition-risk score — trained on real data, not hard-coded heuristics.

## How it works

```mermaid
flowchart LR
    subgraph Sources["Public data sources"]
        FIRMS["NASA FIRMS\nactive-fire labels"]
        OM["Open-Meteo\nweather features"]
        STAC["Sentinel-2 L2A\nsatellite imagery"]
    end

    subgraph Ingest["Ingestion (Phase 1 ✅)"]
        CLI["sentinel ingest CLI"]
    end

    subgraph Store["Storage"]
        PG[("PostGIS\nregions · weather · fires · tiles")]
        S3[("S3 / MinIO\nimage tiles")]
    end

    subgraph ML["Models (Phases 2-3)"]
        CNN["PyTorch CNN\nimagery → risk"]
        LSTM["TensorFlow LSTM\nclimate → risk"]
        ENS["Ensemble\n0-1 risk score"]
    end

    subgraph Serve["Serving (Phases 4-5)"]
        API["FastAPI + Celery"]
        WEB["Next.js map dashboard"]
    end

    FIRMS & OM & STAC --> CLI --> PG
    CLI --> S3
    PG --> CNN & LSTM
    S3 --> CNN
    CNN & LSTM --> ENS --> API --> WEB
```

## Tech stack

| Layer | Tools |
| --- | --- |
| Models | PyTorch (CNN), TensorFlow/Keras (LSTM), scikit-learn |
| Backend | FastAPI, Celery, SQLAlchemy 2.0, GeoAlchemy2 |
| Data | PostgreSQL + PostGIS, Redis, S3 / MinIO |
| Frontend | Next.js + TypeScript, Leaflet/Mapbox |
| Tooling | uv, ruff, mypy, pytest, Docker, GitHub Actions |

## Repository layout

```
sentinel/
├─ .github/workflows/ci.yml      # lint + type-check + tests (PostGIS service)
├─ docker-compose.yml            # PostGIS + Redis + MinIO dev stack
├─ .env.example                  # configuration template (copy to .env)
└─ backend/
   ├─ pyproject.toml             # deps, ruff, mypy, pytest config
   ├─ alembic.ini · migrations/  # schema migrations
   └─ src/sentinel/
      ├─ config.py · logging.py · geo.py
      ├─ db/        # PostGIS ORM models + schema bootstrap
      ├─ ingest/    # FIRMS, Open-Meteo, STAC clients + pipeline
      └─ cli.py     # `sentinel` command-line entry point
```

## Getting started

### 1. Bring up infrastructure

```bash
docker compose up -d        # PostGIS (5432), Redis (6379), MinIO (9000/9001)
cp .env.example .env        # then fill in FIRMS_MAP_KEY (free) — see below
```

### 2. Install the backend

[`uv`](https://docs.astral.sh/uv/) manages the Python toolchain and a pinned, reproducible env:

```bash
cd backend
uv sync --extra dev         # provisions Python 3.12 + all dependencies
```

### 3. Create the schema and ingest data

```bash
uv run sentinel init-db                      # enable PostGIS, create tables
uv run sentinel ingest \
    --region "-120.5,38.5,-120.0,39.0" \     # bbox: min_lon,min_lat,max_lon,max_lat
    --name sierra-nevada \
    --start 2024-08-01 --end 2024-08-10
```

This fetches FIRMS fire detections, Open-Meteo daily weather, and Sentinel-2 scene
thumbnails for the region/date range, uploads the imagery to S3/MinIO, and writes
everything to PostGIS. It prints a summary of rows inserted.

### Data sources

| Source | What | Auth |
| --- | --- | --- |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) | Active-fire detections (labels) | Free `MAP_KEY` |
| [Open-Meteo Archive](https://open-meteo.com/en/docs/historical-weather-api) | Daily temperature, precip, wind, evapotranspiration | None |
| [Sentinel-2 L2A via Earth Search](https://earth-search.aws.element84.com/v1) | Satellite imagery (STAC) | None |

## Development

```bash
cd backend
uv run ruff check .                          # lint
uv run mypy                                   # type-check (strict)
uv run pytest                                 # unit tests (external APIs mocked)
DATABASE_URL=postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel \
  uv run pytest -m integration                # PostGIS round-trip tests
```

CI runs lint, type-check, and the full test suite (including the PostGIS integration
tests against a service container) on every push and pull request.

## Roadmap

- [x] **Phase 1 — Ingestion + PostGIS schema.** FIRMS/weather/imagery clients, geo schema, CLI, CI.
- [ ] **Phase 2 — CNN** on satellite imagery with stratified geospatial cross-validation.
- [ ] **Phase 3 — LSTM** on climate time-series + CNN/LSTM ensemble.
- [ ] **Phase 4 — FastAPI** risk endpoint + Celery scheduled re-scoring.
- [ ] **Phase 5 — Next.js** risk-map dashboard + deployment.
- [ ] **Phase 6 — Hardening + docs** with real measured metrics and a live demo.

## License

MIT
