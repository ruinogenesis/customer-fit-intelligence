"""The app factory: configuration, the API object and its blueprints."""

from collections.abc import Mapping
from typing import Any

from flask import Flask

from cfi_api import health
from cfi_api.errors import CfiApi

API_PREFIX = "/api/v1"

DEFAULTS: dict[str, Any] = {
    "API_TITLE": "Customer Fit Intelligence API",
    "API_VERSION": "v1",
    "OPENAPI_VERSION": "3.1.0",
    "OPENAPI_URL_PREFIX": API_PREFIX,
    "OPENAPI_JSON_PATH": "openapi.json",
    "OPENAPI_SWAGGER_UI_PATH": "docs",
    "OPENAPI_SWAGGER_UI_URL": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/",
    # The live model's version; T202 sets it when it loads model.json.
    "MODEL_VERSION": None,
}


def create_app(config: Mapping[str, Any] | None = None) -> Flask:
    """A new app: DEFAULTS, then config on top (tests pass their own)."""
    app = Flask(__name__)
    app.config.update(DEFAULTS)
    app.config.update(config or {})
    api = CfiApi(app)
    api.register_blueprint(health.blp, url_prefix=API_PREFIX)
    return app
