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
      ├─ data/      # dataset loading + leakage-safe geospatial splitting
      ├─ models/    # ResNet CNN, Keras LSTM, ensemble, inference
      ├─ training/  # training loops, metrics, deterministic seeding
      ├─ serving/   # risk repository + live scorer
      ├─ api/       # FastAPI app (risk endpoints)
      ├─ tasks/     # Celery app + daily re-scoring
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
uv sync --extra dev --extra torch   # Python 3.12 + deps + PyTorch (CPU)
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

### 4. Train the wildfire CNN

```bash
# Fetch the labelled satellite tiles (needs a free Kaggle token in ~/.kaggle).
DATA=$(uv run sentinel prepare-data --source kaggle)

# Transfer-learn a ResNet18 with a leakage-safe geospatial split.
uv run sentinel train-cnn --data-root "$DATA" \
    --output metrics/cnn --limit 5000 --image-size 96 --epochs 6
```

This pools the tiles, re-splits them **geospatially** (no location appears in two
splits), fine-tunes a pretrained ResNet18 with class-weighted loss, evaluates on a
held-out test set, and writes `metrics/cnn/metrics.json` — the single source of
truth for the numbers below.

### 5. Train the climate LSTM + ensemble

```bash
# Fire labels: FPA-FOD US wildfires (Kaggle). Weather: Open-Meteo (no key).
FOD=$(uv run sentinel prepare-data --source kaggle \
      --dataset rtatman/188-million-us-wildfires)/FPA_FOD_20170508.sqlite

uv run sentinel train-lstm --fod-sqlite "$FOD" --output metrics/lstm
uv run sentinel build-ensemble \
    --lstm-dir metrics/lstm \
    --cnn-checkpoint metrics/cnn/cnn_resnet18.pt \
    --output metrics/ensemble
```

`train-lstm` builds region-grouped 14-day weather windows labelled by FOD fires,
trains a class-weighted Keras LSTM on the leakage-safe split, and writes
`metrics/lstm/metrics.json`. `build-ensemble` derives a per-region CNN imagery
prior and fits a logistic meta-model fusing it with the LSTM climate risk.

## Measured results

> Every number here is read straight from a committed `metrics.json`, evaluated
> on **geospatially held-out** regions (never seen in training — no region
> leakage), with all RNG seeded (42). Nothing is hand-tuned for the README.

### Phase 2 — CNN (satellite imagery → fire / no-fire)

| Metric | Test | Notes |
| --- | --- | --- |
| **Recall** | **0.989** | headline — missing a fire is the costly error |
| Precision | 0.863 | |
| F1 | 0.922 | |
| ROC-AUC | 0.981 | |
| Accuracy | 0.915 | |
| False-alarm rate | 0.161 | FP / (FP + TN) |

ResNet18 transfer learning · 5,000 balanced tiles · 402 grid-cell regions
(289 / 57 / 56 train / val / test) · class-weighted loss · CPU, 6 epochs, seed 42.

### Phase 3 — LSTM (climate sequence → ignition in next 7 days)

From `metrics/lstm/metrics.json`. 91,182 sliding 14-day weather windows across
**42 Northern-California grid regions** (30 / 6 / 6 train / val / test, no region
leakage), labelled by FPA-FOD fire occurrences. Predicting ignition **7 days
ahead from weather alone is intentionally hard** — these are honest numbers.

| Metric | Test | Notes |
| --- | --- | --- |
| ROC-AUC | **0.80** | ranking quality on unseen regions |
| Recall | 0.73 | |
| Precision | 0.69 | |
| F1 | 0.71 | |
| False-alarm rate | 0.28 | |

### Phase 3 — CNN + LSTM ensemble (one risk score per region/day)

From `metrics/ensemble/metrics.json`. A logistic meta-model fuses a **static CNN
imagery prior** (one Sentinel-2 tile per region) with the **dynamic LSTM climate
risk**, fit on validation and evaluated on held-out test regions.

| Variant | ROC-AUC | Recall | False-alarm |
| --- | --- | --- | --- |
| **Ensemble (meta-model)** | 0.76 | 0.73 | 0.32 |
| LSTM only | 0.80 | 0.73 | 0.28 |
| CNN only | 0.47 | 0.17 | 0.16 |
| Weighted average (0.5/0.5) | 0.68 | 0.47 | 0.18 |

**Honest finding:** the meta-model learned to weight the LSTM ~3× the CNN
(coefficients 5.18 vs 1.57), because the CNN — trained on Canadian aerial tiles —
**transfers poorly to NorCal Sentinel-2 imagery (domain shift, AUC ≈ 0.47)**. The
climate signal dominates; the imagery prior adds little on this evaluation. The
ensemble plumbing is real and ready for a same-domain imagery model.

## API

A lightweight FastAPI app serves the scores that the Celery worker computes and
stores in PostGIS (the web process stays free of heavy model loading).

```bash
docker compose up -d                    # db, redis, minio, api, worker, beat
# or locally:
cd backend && uv run uvicorn sentinel.api.app:app --reload
```

| Endpoint | Description |
| --- | --- |
| `GET /health` | liveness + version |
| `GET /regions` | tracked regions |
| `GET /risk?region=<id\|name>&date=<YYYY-MM-DD>` | fused risk + component breakdown (latest if no date) |
| `GET /risk/latest` | latest score per region (feeds the map) |

A Celery **beat** schedule re-scores every tracked region daily at 06:00 UTC;
each score fuses the live LSTM climate risk with the CNN imagery prior via the
ensemble meta-model and upserts it into the `risk_scores` table.

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
- [x] **Phase 2 — CNN** on satellite imagery, stratified geospatial split, **98.9% recall** on held-out regions.
- [x] **Phase 3 — LSTM** on climate time-series (AUC **0.80**) + CNN/LSTM ensemble with honest domain-shift analysis.
- [x] **Phase 4 — FastAPI** risk endpoint + Celery daily re-scoring + backend Dockerfile.
- [ ] **Phase 5 — Next.js** risk-map dashboard + deployment.
- [ ] **Phase 6 — Hardening + docs** with real measured metrics and a live demo.

## License

MIT
