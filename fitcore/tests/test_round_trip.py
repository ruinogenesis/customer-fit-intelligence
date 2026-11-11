import os
import uuid

import pytest
from google.cloud import firestore

from fitcore.schemas import load_event

pytestmark = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)


def test_event_round_trips_through_firestore():
    event = load_event(
        {
            "event_id": str(uuid.uuid4()),
            "type": "order_issue",
            "customer_id": "cust_round_trip",
            "occurred_at": "2026-11-11T15:45:00+07:00",
            "schema_version": 1,
            "payload": {"order_id": "ord_001", "issue_type": "cold"},
        }
    )
    event_id = str(event["event_id"])
    expected = {**event, "event_id": event_id}

    db = firestore.Client(project="demo-cfi")
    doc = db.collection("events").document(event_id)
    doc.set(expected)

    assert doc.get().to_dict() == expected
