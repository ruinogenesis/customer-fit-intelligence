# The API's container image for Cloud Run (T204). CI builds it from the repo
# root and deploys it on every merge to main (T208, api/README.md).
# Only what .dockerignore lets in can be copied: fitcore, the API, the model.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# fitcore is not on PyPI, so both packages install from this repo in one
# command, as in CONTRIBUTING.md, with the production web server.
COPY fitcore fitcore
COPY api api
RUN pip install ./fitcore ./api gunicorn==27.0.1

# The model ships inside the image; T310 moves models to Cloud Storage.
COPY notebooks/results/t112_model.json models/t112_model.json
ENV CFI_MODEL_PATH=/app/models/t112_model.json \
    CFI_MODEL_VERSION=t112-sim

# Run as an ordinary user, not root.
RUN useradd --system --uid 10001 cfi
USER cfi

# Cloud Run sends requests to $PORT (8080 unless it says otherwise). One
# process with eight threads; Cloud Run adds instances when it needs more.
# exec makes gunicorn the main process, so it gets Cloud Run's SIGTERM and
# shuts down cleanly. A container needs no gunicorn control socket.
CMD exec gunicorn --bind ":${PORT:-8080}" --workers 1 --threads 8 --timeout 0 \
    --no-control-socket "cfi_api:create_app()"
