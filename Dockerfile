# The API's container image for Cloud Run (T204). CI builds it from the repo
# root and deploys it on every merge to main (T208, api/README.md).
# Only what .dockerignore lets in can be copied: fitcore and the API. The
# model comes from the registry in Cloud Storage (T310), or from a file
# named by CFI_MODEL_PATH, as CI's container check mounts one.
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

# Run as an ordinary user, not root.
RUN useradd --system --uid 10001 cfi
USER cfi

# Cloud Run sends requests to $PORT (8080 unless it says otherwise). One
# process with eight threads; Cloud Run adds instances when it needs more.
# exec makes gunicorn the main process, so it gets Cloud Run's SIGTERM and
# shuts down cleanly. A container needs no gunicorn control socket.
# APP_MODULE picks the app: the API by default, or the rules service
# (cfi-rules, T501) with APP_MODULE="cfi_api.rules_service:create_rules_app()".
CMD exec gunicorn --bind ":${PORT:-8080}" --workers 1 --threads 8 --timeout 0 \
    --no-control-socket "${APP_MODULE:-cfi_api:create_app()}"
