"""The rules service (T501): Pub/Sub pushes each new score entry here.

It runs as its own private Cloud Run service, cfi-rules, from the API's
image (APP_MODULE in the Dockerfile). Cloud Run lets in only the push
subscription's and the sweep's service accounts, so this app checks no
tokens itself. The
subscription takes only score entries (the messages' collection attribute,
cfi_api.export) from the topic every saved document goes to.

POST /push takes Pub/Sub's push envelope, whose data is one row of
raw.changes:

- a score entry: the engine decides (cfi_api.engine), then 204. If deciding
  fails, 500, and Pub/Sub sends it again: deciding again logs nothing twice.
- any other document: 204, nothing done.
- anything that is not such a message: 204, logged as an error, and dropped,
  since sending it again could never make it one.

POST /sweep runs the daily sweep (T505; cfi_api.engine.sweep): Cloud
Scheduler calls it each morning as its own service account, which Cloud Run
lets in too. 200 with {"decided": n}; if sweeping fails, 500, and Scheduler
tries again: sweeping again logs nothing twice.

Settings, as the API's (cfi_api.app): STORE ("memory" or "firestore") and
PROJECT, from CFI_* environment variables or config.
"""

import base64
import binascii
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from flask import Flask, current_app, request

from cfi_api.engine import Automations, Scoring, decide, sweep
from cfi_api.runtime import make_store

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {"STORE": "memory", "PROJECT": None}


class NotAScoring(ValueError):
    """The push is not a Pub/Sub message holding a row of raw.changes."""


def change(envelope: Any) -> dict[str, Any]:
    """The row of raw.changes inside a push envelope. Raises NotAScoring."""
    try:
        data = base64.b64decode(envelope["message"]["data"], validate=True)
        row = json.loads(data)
        if not isinstance(row, dict) or not isinstance(row.get("collection"), str):
            raise NotAScoring("no collection")
        return row
    except (TypeError, KeyError, ValueError, binascii.Error) as err:
        raise NotAScoring(str(err)) from err


def scoring(row: Mapping[str, Any]) -> Scoring:
    """The score entry in a raw.changes row. Raises NotAScoring."""
    try:
        parts = row["path"].split("/")
        if len(parts) != 4 or parts[0] != "customers" or parts[2] != "scores":
            raise NotAScoring(f"not a score entry's path: {row['path']!r}")
        entry = json.loads(row["document"])
        return Scoring(customer_id=parts[1], event_id=parts[3], entry=entry)
    except (TypeError, KeyError, ValueError, AttributeError) as err:
        raise NotAScoring(str(err)) from err


def push() -> tuple[str, int]:
    try:
        row = change(request.get_json(silent=True))
        if row["collection"] != "scores":
            return "", 204
        new = scoring(row)
    except NotAScoring as err:
        logger.error("Dropped a push that is not a change message: %s", err)
        return "", 204
    store: Automations = current_app.extensions["cfi_store"]
    _logged(decide(store, new, datetime.now(UTC)))
    return "", 204


def sweep_now() -> tuple[dict[str, int], int]:
    store: Automations = current_app.extensions["cfi_store"]
    decided = sweep(store, datetime.now(UTC))
    _logged(decided)
    logger.info("The daily sweep decided %d", len(decided))
    return {"decided": len(decided)}, 200


def _logged(decided: list[dict[str, Any]]) -> None:
    for action in decided:
        logger.info(
            "%s decided for %s (%s, %s)",
            action["rule_id"],
            action["customer_id"],
            action["arm"],
            "dry run" if action["dry_run"] else action["channel"],
        )


def create_rules_app(config: Mapping[str, Any] | None = None) -> Flask:
    """A new rules service: DEFAULTS, then CFI_* environment variables, then
    config. Raises ValueError for an unknown store."""
    # Decisions are logged at INFO; gunicorn sets up only its own loggers.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    app = Flask(__name__, static_folder=None)
    app.config.update(DEFAULTS)
    app.config.from_prefixed_env("CFI")
    app.config.update(config or {})
    app.extensions["cfi_store"] = make_store(app.config["STORE"], app.config["PROJECT"])
    app.add_url_rule("/push", "push", push, methods=["POST"])
    app.add_url_rule("/sweep", "sweep", sweep_now, methods=["POST"])
    return app
