# syntax=docker/dockerfile:1
#
# ---------------------------------------------------------------------------
# SIH 2026 / Problem SIH26072 -- Thunderstorm Nowcast Decision Support (VOTV)
#
# Production image for the existing Phase 8 Flask application.
#
# Deployment only: this image serves the artefact that was already trained and
# verified. It does not train, retrain, tune, or re-threshold anything, and it
# does not reference radar, satellite, lightning, NWP or multi-cell data.
# models/ and outputs/ are copied read-only and are never written to.
# ---------------------------------------------------------------------------

# Python 3.12 matches the interpreter the Phase 6 model was trained and
# verified with (3.12.10, recorded in models/thunderstorm_nowcast_1h_metadata.json),
# so the joblib artifact unpickles into the environment it was written from.
# Debian bookworm (glibc 2.36) satisfies the manylinux_2_28 wheels that
# pandas/scipy ship, and slim is enough because every dependency has a
# prebuilt manylinux cp312 wheel -- no compiler is needed in the image.
FROM python:3.12-slim-bookworm

# PYTHONDONTWRITEBYTECODE: no .pyc writes inside the container.
# PYTHONUNBUFFERED: logs reach the platform log stream immediately.
# PORT: safe local default; Render injects its own value at runtime and the
# start command below always honours it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=10000

WORKDIR /app

# Non-root runtime user, created before any COPY so the application layers can
# be copied with the right ownership in one pass (no recursive chown layer,
# which would duplicate the ~800 MB of model and data files).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Dependencies first, so an application-only change reuses this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

# Application, model artifacts, the Phase 4 feature module, and the read-only
# evidence the API serves. .dockerignore trims the build context.
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 10000

# Bind all interfaces on the port the platform provides (Render sets PORT),
# falling back to 10000 locally.
#
# --preload imports app.py once in the master process, so the 1-hour nowcast
#   bundle and the legacy surrogate bundle are read a single time and shared
#   with forked workers instead of being loaded per worker.
# --workers 1 --threads 4 keeps exactly one copy of those models resident. The
#   service is I/O bound (it waits on the upstream Open-Meteo call), not CPU
#   bound, so threads give concurrency without duplicating model memory.
# --timeout 120 covers a slow upstream weather response (the engine's own HTTP
#   timeout is 20 s) plus feature construction, well above gunicorn's 30 s
#   default that a cold first request could otherwise trip.
# --access-logfile -/--error-logfile - send the logs to stdout/stderr so the
#   platform's log stream shows them.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 --graceful-timeout 30 --preload --forwarded-allow-ips='*' --access-logfile - --error-logfile - app:app"]
