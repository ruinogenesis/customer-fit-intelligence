"""The app factory: configuration, the model, the store and the blueprints."""

from collections.abc import Mapping
from typing import Any

from flask import Flask

from cfi_api import events, health, score
from cfi_api.errors import CfiApi
from cfi_api.runtime import load_model, make_store
from cfi_api.write_key import SCHEME, SECURITY_SCHEME, require_write_key

API_PREFIX = "/api/v1"

DEFAULTS: dict[str, Any] = {
    "API_TITLE": "Customer Fit Intelligence API",
    "API_VERSION": "v1",
    "OPENAPI_VERSION": "3.1.0",
    "OPENAPI_URL_PREFIX": API_PREFIX,
    "OPENAPI_JSON_PATH": "openapi.json",
    "OPENAPI_SWAGGER_UI_PATH": "docs",
    "OPENAPI_SWAGGER_UI_URL": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/",
    # The model.json to score with and the version to report: set both or
    # neither. With neither, /health reports no model and /events answers 503.
    "MODEL_PATH": None,
    "MODEL_VERSION": None,
    # Where events and state live: "memory" (this process only) or "firestore".
    "STORE": "memory",
    # The Google Cloud project for Firestore; None uses the environment's.
    "FIRESTORE_PROJECT": None,
    # The key every request but a read must carry as X-CFI-Key (T205, until
    # T303); None checks nothing.
    "WRITE_KEY": None,
}


def create_app(config: Mapping[str, Any] | None = None) -> Flask:
    """A new app: DEFAULTS, then CFI_* environment variables, then config.

    An environment variable drops its prefix: CFI_MODEL_PATH sets
    MODEL_PATH. Tests pass their own config. Raises ValueError for a model
    with only a path or only a version, an unknown store or an empty write
    key.
    """
    app = Flask(__name__)
    app.config.update(DEFAULTS)
    app.config.from_prefixed_env("CFI")
    app.config.update(config or {})
    app.extensions["cfi_model"] = load_model(
        app.config["MODEL_PATH"], app.config["MODEL_VERSION"]
    )
    app.extensions["cfi_store"] = make_store(
        app.config["STORE"], app.config["FIRESTORE_PROJECT"]
    )
    require_write_key(app, app.config["WRITE_KEY"])
    api = CfiApi(app)
    api.spec.components.security_scheme(SCHEME, SECURITY_SCHEME)
    api.register_blueprint(health.blp, url_prefix=API_PREFIX)
    api.register_blueprint(events.blp, url_prefix=API_PREFIX)
    api.register_blueprint(score.blp, url_prefix=API_PREFIX)
    return app
