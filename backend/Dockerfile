# Backend image — serves the FastAPI app and runs the Celery worker/beat.
# Build context: ./backend
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# libgomp1 is required by torch / scikit-learn; curl for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra torch --extra tf --frozen --no-install-project

# Then the project source.
COPY . .
RUN uv sync --extra torch --extra tf --frozen

ENV PATH="/app/.venv/bin:${PATH}"

# Run as a non-root user.
RUN useradd --system --uid 1001 sentinel \
    && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default command runs the API; the worker/beat override this in compose.
CMD ["uvicorn", "sentinel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
