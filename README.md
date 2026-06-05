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
├─ .github/workflows/ci.yml      # lint + type-check + tests (backend + frontend)
├─ docker-compose.yml            # full local stack
├─ render.yaml                   # backend deploy blueprint (Render)
├─ backend/
│  ├─ pyproject.toml · Dockerfile · alembic.ini · migrations/
│  └─ src/sentinel/
│     ├─ config.py · logging.py · geo.py
│     ├─ db/        # PostGIS ORM models + schema bootstrap
│     ├─ ingest/    # FIRMS, Open-Meteo, STAC clients + pipeline
│     ├─ data/      # dataset loading + leakage-safe geospatial splitting
│     ├─ models/    # ResNet CNN, Keras LSTM, ensemble, inference
│     ├─ training/  # training loops, metrics, deterministic seeding
│     ├─ serving/   # risk repository + live scorer
│     ├─ api/       # FastAPI app (risk endpoints)
│     ├─ tasks/     # Celery app + daily re-scoring
│     └─ cli.py     # `sentinel` command-line entry point
└─ frontend/        # Next.js + TypeScript + Leaflet dashboard
   ├─ Dockerfile
   └─ src/{app,components,lib}
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

### 5. Train the climate LSTM + ensemble (global)

```bash
# Global fire labels: NASA MODIS active fire, 1 year (Kaggle). Weather: Open-Meteo.
GF=$(uv run sentinel prepare-data --source kaggle \
     --dataset brsdincer/leo-satellite-global-fire-data-1-year-nasa)
CSV="$GF/WORLD/2020_2021_01_JUN_TO_FROM_MODIS.csv"

uv run sentinel train-lstm-global --fire-csv "$CSV" --output metrics/lstm \
    --max-regions 200 --cell-size 2.0
uv run sentinel build-ensemble \
    --lstm-dir metrics/lstm \
    --cnn-checkpoint metrics/cnn/cnn_resnet18.pt \
    --output metrics/ensemble
```

`train-lstm-global` selects the most fire-active grid cells **with geographic
spread** (round-robin across macro-cells so every continent is represented),
fetches a year of Open-Meteo weather per region, builds 14-day windows on the
leakage-safe split, trains a class-weighted Keras LSTM, and writes
`metrics/lstm/metrics.json`. `build-ensemble` derives a per-region CNN imagery
prior and fits a logistic meta-model fusing it with the LSTM climate risk.
(For a US-only model instead, `train-lstm --fod-sqlite <FPA-FOD.sqlite>`.)

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

### Phase 3 — LSTM (climate sequence → ignition in next 7 days), **global**

From `metrics/lstm/metrics.json`. 72,200 sliding 14-day weather windows across
**200 fire-active regions worldwide** (142 / 29 / 29 train / val / test, no region
leakage — held-out regions span multiple continents), labelled by NASA MODIS
global active-fire detections (Jun 2020 – Jun 2021). Predicting ignition **7 days
ahead from weather alone, evaluated on entirely unseen parts of the planet** —
these are honest numbers.

| Metric | Test | Notes |
| --- | --- | --- |
| ROC-AUC | **0.85** | ranking quality on unseen global regions |
| Recall | 0.85 | |
| Precision | 0.86 | |
| F1 | 0.85 | |
| False-alarm rate | 0.33 | |

> The earlier US-only model (FPA-FOD, Northern California, AUC 0.80) was retrained
> on a globally-distributed fire dataset — region selection spreads across
> continents so the model sees savanna, Mediterranean, boreal, and temperate fire
> regimes, not just one. Any point on Earth is scorable via `/risk/point`.

### Phase 3 — CNN + LSTM ensemble (one risk score per region/day)

From `metrics/ensemble/metrics.json`. A logistic meta-model fuses a **static CNN
imagery prior** (one Sentinel-2 tile per region) with the **dynamic LSTM climate
risk**, fit on validation and evaluated on the 29 held-out global regions.

| Variant | ROC-AUC | Recall | False-alarm |
| --- | --- | --- | --- |
| LSTM only | 0.854 | 0.847 | 0.335 |
| **Ensemble (meta-model)** | 0.826 | 0.805 | **0.287** |
| CNN only | 0.620 | 0.168 | 0.065 |
| Weighted average (0.5/0.5) | 0.844 | 0.362 | 0.073 |

**Honest finding:** the climate LSTM is the dominant signal (AUC 0.85). The
Canada-trained CNN carries *some* worldwide signal (AUC 0.62 — flammable
landscape correlates with fire), but it's noisy, so the meta-model down-weights
it (it even learns a slightly negative CNN coefficient). The fusion trades a
little ranking AUC for a meaningfully **lower false-alarm rate (0.287 vs 0.335)**
than the LSTM alone — a real, if modest, multimodal benefit, reported straight.

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

## Dashboard

A Next.js + TypeScript dashboard (`frontend/`) renders a Leaflet map with
color-coded regional risk overlays, a region detail panel breaking down the
ensemble / climate / imagery component scores, and a date selector — in a clean
white + ember "fire" theme (Bricolage Grotesque + IBM Plex, lucide icons).

```bash
cd frontend
pnpm install
cp .env.example .env.local          # point NEXT_PUBLIC_API_URL at the API
pnpm dev                             # http://localhost:3000
```

Populate the map with regions (and, with `--score`, real fused scores):

```bash
cd backend
uv run sentinel seed-demo --score    # grid regions over N. California
```

## Deploy

- **Frontend → Vercel.** Import the repo, set the root to `frontend/`, and add
  `NEXT_PUBLIC_API_URL`. Vercel auto-detects Next.js.
- **Backend → Render.** [`render.yaml`](render.yaml) is a Blueprint that
  provisions PostGIS, Redis, the API (Docker), and the Celery worker/beat. Set
  `CORS_ORIGINS` to your Vercel URL. (`docker-compose.yml` runs the whole stack
  locally.)

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
tests against a service container) on every push and pull request — for both the
backend and the frontend.

## Security

- **No secrets in code.** Configuration is read from `.env` (only `.env.example`
  is committed); the Kaggle token lives in `~/.kaggle`, never the repo.
- **Input validation** on every boundary: bbox/date parsing, Pydantic settings,
  cloud-cover bounds.
- **No SQL injection surface** — all queries use SQLAlchemy's parameterized ORM;
  no string-built SQL with user input.
- **Least privilege containers** — backend and frontend images run as non-root
  users with pinned, lockfile-driven dependencies (`uv.lock`, `pnpm-lock.yaml`).
- **Restricted CORS** — the API only allows the configured `CORS_ORIGINS`, and
  exposes read-only `GET` endpoints.

## Demo & screenshots

The dashboard is data-driven, so populate it first, then capture:

```bash
docker compose up -d                     # full stack
cd backend && uv run sentinel seed-demo --score   # regions + real fused scores
cd ../frontend && pnpm dev               # open http://localhost:3000
```

> Live demo URL and screenshots are added after deploying to Vercel + Render
> (see **Deploy** above) — the configs are turnkey.

## Roadmap

- [x] **Phase 1 — Ingestion + PostGIS schema.** FIRMS/weather/imagery clients, geo schema, CLI, CI.
- [x] **Phase 2 — CNN** on satellite imagery, stratified geospatial split, **98.9% recall** on held-out regions.
- [x] **Phase 3 — LSTM** on climate time-series (AUC **0.80**) + CNN/LSTM ensemble with honest domain-shift analysis.
- [x] **Phase 4 — FastAPI** risk endpoint + Celery daily re-scoring + backend Dockerfile.
- [x] **Phase 5 — Next.js** Leaflet risk-map dashboard + Vercel/Render deploy config.
- [x] **Phase 6 — Hardening + docs**: expanded tests, non-root images, security pass, full README. *(Live demo URL + screenshots pending deploy.)*

## License

MIT
