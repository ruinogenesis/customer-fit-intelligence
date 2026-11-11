import pytest
from marshmallow import ValidationError

from fitcore.schemas import EVENT_TYPES, PAYLOAD_SCHEMAS, EnvelopeSchema, load_event


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


VALID_PAYLOADS = {
    "order_placed": {
        "order_id": "ord_001",
        "restaurant_id": "rest_042",
        "basket_value": 35000,
        "fees": 4000,
        "promised_eta": "2026-11-08T20:05:00+07:00",
        "promo_used": False,
    },
    "order_delivered": {"order_id": "ord_001", "courier_id": "cour_007"},
    "order_issue": {"order_id": "ord_001", "issue_type": "cold"},
    "rating_submitted": {"order_id": "ord_001", "food": 4, "courier": 5},
    "app_session": {
        "session_id": "sess_123",
        "time_to_checkout_s": 95,
        "steps": 6,
        "errors": 0,
    },
    "recommendation_shown": {
        "recommendation_id": "rec_9",
        "items": ["dish_1", "dish_2"],
    },
    "recommendation_clicked": {
        "recommendation_id": "rec_9",
        "item_id": "dish_2",
        "position": 1,
    },
    "support_case": {
        "case_id": "case_5",
        "order_id": "ord_001",
        "opened_at": "2026-11-08T20:30:00+07:00",
        "refund_amount": 8000,
        "satisfaction": 4,
    },
    "survey_response": {"item_id": "love_miss_app", "value": 6},
    "action_sent": {
        "action_id": "act_1",
        "rule_id": "service_recovery",
        "arm": "treatment",
        "channel": "push",
    },
    "action_outcome": {"action_id": "act_1", "result": "opened"},
}


def test_every_event_type_has_a_sample():
    assert set(VALID_PAYLOADS) == set(EVENT_TYPES)


@pytest.mark.parametrize("event_type", sorted(VALID_PAYLOADS))
def test_valid_payload_loads(event_type):
    PAYLOAD_SCHEMAS[event_type]().load(VALID_PAYLOADS[event_type])


@pytest.mark.parametrize(
    ("event_type", "changes", "bad_field"),
    [
        ("order_placed", {"basket_value": 350.0}, "basket_value"),  # not whole satang
        ("order_placed", {"basket_value": -1}, "basket_value"),
        ("order_issue", {"issue_type": "rude"}, "issue_type"),
        ("rating_submitted", {"food": 6}, "food"),
        ("survey_response", {"value": 0}, "value"),
        ("action_sent", {"arm": "maybe"}, "arm"),
        ("recommendation_shown", {"items": []}, "items"),
        ("order_delivered", {"tip": 2000}, "tip"),  # unknown field
    ],
)
def test_malformed_payload_is_rejected(event_type, changes, bad_field):
    raw = {**VALID_PAYLOADS[event_type], **changes}
    with pytest.raises(ValidationError) as excinfo:
        PAYLOAD_SCHEMAS[event_type]().load(raw)
    assert bad_field in excinfo.value.messages


def test_rating_needs_at_least_one_score():
    with pytest.raises(ValidationError) as excinfo:
        PAYLOAD_SCHEMAS["rating_submitted"]().load({"order_id": "ord_001"})
    assert "_schema" in excinfo.value.messages


def test_session_without_checkout_is_allowed():
    raw = {**VALID_PAYLOADS["app_session"], "time_to_checkout_s": None}
    PAYLOAD_SCHEMAS["app_session"]().load(raw)


def make_event(event_type, payload):
    return {**valid_envelope(), "type": event_type, "payload": payload}


@pytest.mark.parametrize("event_type", sorted(VALID_PAYLOADS))
def test_load_event_accepts_every_valid_sample(event_type):
    event = load_event(make_event(event_type, VALID_PAYLOADS[event_type]))
    assert event["type"] == event_type


def test_load_event_nests_payload_errors_under_payload():
    raw = make_event("order_issue", {"order_id": "ord_001", "issue_type": "rude"})
    with pytest.raises(ValidationError) as excinfo:
        load_event(raw)
    assert "issue_type" in excinfo.value.messages["payload"]


def test_load_event_checks_payload_against_its_own_type():
    raw = make_event("order_placed", VALID_PAYLOADS["order_delivered"])
    with pytest.raises(ValidationError) as excinfo:
        load_event(raw)
    assert "payload" in excinfo.value.messages


def test_payload_timestamps_are_converted_to_utc():
    event = load_event(make_event("order_placed", VALID_PAYLOADS["order_placed"]))
    assert event["payload"]["promised_eta"].isoformat() == "2026-11-08T13:05:00+00:00"
