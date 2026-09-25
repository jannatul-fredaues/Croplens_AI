# Build from the REPO ROOT, not from api/, so this can COPY models/:
#   docker build -f api/Dockerfile -t croplens-api .
#   (or `make docker-build`)
#
# Multi-stage: build wheels in one layer, copy only the installed
# packages into the final image, so pip's build cache doesn't bloat it.

# ---------- Stage 1: build ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Needed to build a couple of Python packages from source on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim

WORKDIR /app

# curl is used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

# Installed packages from the build stage.
COPY --from=builder /install /usr/local

# Application code and versioned artifacts.
COPY api/app ./app
COPY api/data ./data
COPY models ./models

# Absolute paths inside the container (overrides the relative dev paths
# in .env.example, which only make sense when running from api/ locally).
ENV APP_ENV=production \
    MODEL_PATH=/app/models/v1/croplens_v1.keras \
    LABELS_PATH=/app/models/v1/labels.json \
    CROP_INFO_PATH=/app/data/crop_info.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Real values (CONFIDENCE_THRESHOLD, ALLOWED_ORIGINS, MODEL_VERSION, etc.)
# are set as environment variables in the host dashboard (Render/Cloud Run),
# not baked into the image - see .env.example for the full list.

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form so ${PORT} is expanded - Cloud Run/Render inject PORT at runtime,
# and it may differ from the 8000 used for local `docker run`.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
