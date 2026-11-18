"""T305: Firestore deletes each raw event 30 days after the API received it.

events/{event_id} carries expires_at, a timestamp: when the API received the
event plus EVENT_RETENTION. firestore.indexes.json makes expires_at the
collection's TTL field and leaves every field of an events document out of
the indexes, since nothing queries them. BigQuery keeps every event (T306).
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from flask import Flask
from test_events import URL, memory, placed

from cfi_api import create_app
from cfi_api.store import EVENT_RETENTION, EXPIRES_AT

INDEXES = Path(__file__).parents[2] / "firestore.indexes.json"


@pytest.fixture
def app(reference_model: str) -> Flask:
    return create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})


def _stored(app: Flask, event_id: str) -> dict[str, Any]:
    return memory(app).documents[f"events/{event_id}"]


def test_an_event_expires_30_days_after_the_api_received_it(app: Flask) -> None:
    # Dated months back, as a fake run's events are: the clock is the API's.
    event = {**placed(), "occurred_at": "2026-01-01T12:00:00+00:00"}
    before = datetime.now(UTC)
    assert app.test_client().post(URL, json=event).status_code == 201
    after = datetime.now(UTC)
    expires = _stored(app, event["event_id"])[EXPIRES_AT]
    assert EVENT_RETENTION == timedelta(days=30)
    assert isinstance(expires, datetime) and expires.utcoffset() == timedelta(0)
    assert before + EVENT_RETENTION <= expires <= after + EVENT_RETENTION


def test_a_repeat_saves_nothing_so_its_expiry_stays(app: Flask) -> None:
    client, event = app.test_client(), placed()
    assert client.post(URL, json=event).status_code == 201
    first = _stored(app, event["event_id"])[EXPIRES_AT]
    assert client.post(URL, json=event).status_code == 200
    assert _stored(app, event["event_id"])[EXPIRES_AT] == first


def test_the_ttl_policy_is_on_the_expiry_and_no_events_field_is_indexed(
    app: Flask,
) -> None:
    event = placed()
    assert app.test_client().post(URL, json=event).status_code == 201
    fields = set(_stored(app, event["event_id"]))
    overrides = [
        override
        for override in json.loads(INDEXES.read_text())["fieldOverrides"]
        if override["collectionGroup"] == "events"
    ]
    # Every field an events document holds has a decision, and none is indexed.
    assert sorted(o["fieldPath"] for o in overrides) == sorted(fields)
    assert all(override["indexes"] == [] for override in overrides)
    assert [o["fieldPath"] for o in overrides if o.get("ttl")] == [EXPIRES_AT]
    assert all(o.get("ttl") in (None, True) for o in overrides)


def test_the_docs_say_how_long_a_repeat_is_recognised(app: Flask) -> None:
    spec = app.test_client().get("/api/v1/openapi.json").get_json()
    description = spec["paths"]["/api/v1/events"]["post"]["description"]
    assert f"{EVENT_RETENTION.days} days" in description
