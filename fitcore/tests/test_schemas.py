import pytest
from marshmallow import ValidationError

from fitcore.schemas import EnvelopeSchema


def valid_envelope():
    return {
        "event_id": "6f1c2a9e-3b4d-4c8e-9f10-2a3b4c5d6e7f",
        "type": "order_placed",
        "customer_id": "cust_001",
        "occurred_at": "2026-11-08T19:30:00+07:00",
        "schema_version": 1,
        "payload": {},
    }


def test_valid_envelope_loads_and_converts_to_utc():
    event = EnvelopeSchema().load(valid_envelope())
    assert event["occurred_at"].isoformat() == "2026-11-08T12:30:00+00:00"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("event_id", "not-a-uuid"),
        ("type", "order_teleported"),
        ("customer_id", ""),
        ("occurred_at", "2026-11-08T19:30:00"),  # no timezone
        ("schema_version", 2),
        ("schema_version", "1"),  # text, not a number
    ],
)
def test_malformed_envelope_is_rejected(field, bad_value):
    raw = valid_envelope()
    raw[field] = bad_value
    with pytest.raises(ValidationError) as excinfo:
        EnvelopeSchema().load(raw)
    assert field in excinfo.value.messages


def test_missing_field_is_rejected():
    raw = valid_envelope()
    del raw["customer_id"]
    with pytest.raises(ValidationError) as excinfo:
        EnvelopeSchema().load(raw)
    assert "customer_id" in excinfo.value.messages


def test_unknown_field_is_rejected():
    raw = valid_envelope()
    raw["colour"] = "blue"
    with pytest.raises(ValidationError) as excinfo:
        EnvelopeSchema().load(raw)
    assert "colour" in excinfo.value.messages
