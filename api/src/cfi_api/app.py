"""The app factory: configuration, the model, the store, the publisher, the
history and the blueprints."""

from collections.abc import Mapping
from typing import Any

from flask import Flask

from cfi_api import customers, drivers, events, health, matrix, models, score
from cfi_api.access import SECURITY_SCHEMES, require_access
from cfi_api.errors import CfiApi
from cfi_api.runtime import (
    load_model,
    make_history,
    make_models,
    make_publisher,
    make_store,
)

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
    # Or the model registry (cfi_api.registry, T310): "gcs" scores with the
    # store's live version, its files in Cloud Storage; "memory" keeps the
    # files in this process (tests). Not with MODEL_PATH.
    "REGISTRY": None,
    # Where events and state live: "memory" (this process only) or "firestore".
    "STORE": "memory",
    # The Google Cloud project, for Firestore and sign-in tokens; None uses
    # the environment's.
    "PROJECT": None,
    # Who may call what (cfi_api.access): "firebase" checks staff sign-in
    # tokens and the integration key; None checks nothing.
    "AUTH": None,
    # The integration key that opens POST /events as X-CFI-Key; needs AUTH.
    "WRITE_KEY": None,
    # The Pub/Sub topic that takes every saved document on to BigQuery
    # (cfi_api.export, T306); needs PROJECT. None exports nothing.
    "TOPIC": None,
    # Where /matrix reads movement between two dates (cfi_api.history, T313):
    # "bigquery", in PROJECT. None reads no history: movement answers 503.
    "HISTORY": None,
}


def create_app(config: Mapping[str, Any] | None = None) -> Flask:
    """A new app: DEFAULTS, then CFI_* environment variables, then config.

    An environment variable drops its prefix: CFI_MODEL_PATH sets
    MODEL_PATH. Tests pass their own config. Raises ValueError for a model
    with only a path or only a version, a model file and a registry both,
    an unknown store or registry, an unknown AUTH, a write key that would
    not be checked, a topic without a project or an unknown history.
    """
    # No static files: the API serves only its endpoints.
    app = Flask(__name__, static_folder=None)
    app.config.update(DEFAULTS)
    app.config.from_prefixed_env("CFI")
    app.config.update(config or {})
    app.extensions["cfi_model"] = load_model(
        app.config["MODEL_PATH"], app.config["MODEL_VERSION"]
    )
    if app.config["REGISTRY"] is not None and app.extensions["cfi_model"]:
        raise ValueError("set MODEL_PATH or REGISTRY, not both")
    app.extensions["cfi_models"] = make_models(
        app.config["REGISTRY"], app.config["PROJECT"]
    )
    publisher = make_publisher(app.config["TOPIC"], app.config["PROJECT"])
    app.extensions["cfi_publisher"] = publisher
    app.extensions["cfi_store"] = make_store(
        app.config["STORE"], app.config["PROJECT"], outbox=publisher is not None
    )
    app.extensions["cfi_history"] = make_history(
        app.config["HISTORY"], app.config["PROJECT"]
    )
    require_access(
        app, app.config["AUTH"], app.config["PROJECT"], app.config["WRITE_KEY"]
    )
    api = CfiApi(app)
    for name, scheme in SECURITY_SCHEMES.items():
        api.spec.components.security_scheme(name, scheme)
    api.register_blueprint(health.blp, url_prefix=API_PREFIX)
    api.register_blueprint(events.blp, url_prefix=API_PREFIX)
    api.register_blueprint(score.blp, url_prefix=API_PREFIX)
    api.register_blueprint(customers.blp, url_prefix=API_PREFIX)
    api.register_blueprint(matrix.blp, url_prefix=API_PREFIX)
    api.register_blueprint(drivers.blp, url_prefix=API_PREFIX)
    api.register_blueprint(models.blp, url_prefix=API_PREFIX)
    return app
