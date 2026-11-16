"""Event schemas v1: the contract for every event that enters the system."""

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from marshmallow import (
    RAISE,
    Schema,
    ValidationError,
    fields,
    post_load,
    validate,
    validates_schema,
)

SCHEMA_VERSION = 1

# Choices that events and stored documents (fitcore.documents) share.
ISSUE_TYPES: tuple[str, ...] = ("missing", "wrong", "cold", "late")
RATINGS: tuple[str, ...] = ("food", "courier", "accuracy", "overall")
ARMS: tuple[str, ...] = ("treatment", "holdout")
CHANNELS: tuple[str, ...] = ("push", "in_app", "email", "voucher")
ACTION_RESULTS: tuple[str, ...] = (
    "delivered",
    "opened",
    "clicked",
    "redeemed",
    "failed",
)


def check_id(value: str) -> None:
    """Reject an id that cannot be a Firestore document id.

    No "/" (it would point into another collection), not "." or "..", not
    __name__, and at most 1,500 bytes.
    """
    if (
        "/" in value
        or value in (".", "..")
        or re.fullmatch("__.*__", value)
        or len(value.encode()) > 1500
    ):
        raise ValidationError(
            "Not a usable id: no '/', '.', '..', __name__ or 1,500+ bytes."
        )


def _id(required: bool = True) -> fields.String:
    return fields.String(required=required, validate=[validate.Length(min=1), check_id])


def _count() -> fields.Integer:
    """A whole number of 0 or more: money in satang, counts, positions."""
    return fields.Integer(required=True, strict=True, validate=validate.Range(min=0))


def _score(low: int, high: int, required: bool = True) -> fields.Integer:
    return fields.Integer(
        required=required, strict=True, validate=validate.Range(min=low, max=high)
    )


def _one_of(*choices: str) -> fields.String:
    return fields.String(required=True, validate=validate.OneOf(choices))


class StrictSchema(Schema):
    """Base for every schema in this module: unknown fields are rejected."""

    class Meta:
        unknown = RAISE


class StrictBoolean(fields.Boolean):
    """true or false only: not 1, 0 or "true".

    (fields.Boolean with truthy={True} still lets 1 in, because 1 == True.)
    """

    def _deserialize(
        self,
        value: Any,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> bool:
        if not isinstance(value, bool):
            raise self.make_error("invalid", input=value)
        return value


class UtcDateTime(fields.AwareDateTime):
    """A timestamp that must carry a timezone; always returned in UTC."""

    def _deserialize(
        self,
        value: Any,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> datetime:
        return super()._deserialize(value, attr, data, **kwargs).astimezone(UTC)


# --- Payloads: one schema per event type ---


class OrderPlacedPayload(StrictSchema):
    order_id = _id()
    restaurant_id = _id()
    basket_value = _count()
    fees = _count()
    promised_eta = UtcDateTime(required=True)
    promo_used = StrictBoolean(required=True)


class OrderDeliveredPayload(StrictSchema):
    order_id = _id()
    courier_id = _id()


class OrderIssuePayload(StrictSchema):
    order_id = _id()
    issue_type = _one_of(*ISSUE_TYPES)


class RatingSubmittedPayload(StrictSchema):
    order_id = _id()
    food = _score(1, 5, required=False)
    courier = _score(1, 5, required=False)
    accuracy = _score(1, 5, required=False)
    overall = _score(1, 5, required=False)

    @validates_schema
    def at_least_one_score(self, data: dict[str, Any], **kwargs: Any) -> None:
        if not any(name in data for name in RATINGS):
            raise ValidationError("At least one score is required.")


class AppSessionPayload(StrictSchema):
    session_id = _id()
    time_to_checkout_s = fields.Integer(
        required=True, strict=True, allow_none=True, validate=validate.Range(min=0)
    )
    steps = _count()
    errors = _count()


class RecommendationShownPayload(StrictSchema):
    recommendation_id = _id()
    items = fields.List(
        fields.String(validate=validate.Length(min=1)),
        required=True,
        validate=validate.Length(min=1),
    )


class RecommendationClickedPayload(StrictSchema):
    recommendation_id = _id()
    item_id = _id()
    position = _count()


class SupportCasePayload(StrictSchema):
    case_id = _id()
    order_id = _id(required=False)
    opened_at = UtcDateTime(required=True)
    refund_amount = _count()
    satisfaction = _score(1, 5, required=False)


class SurveyResponsePayload(StrictSchema):
    item_id = _id()
    value = _score(1, 7)


class ActionSentPayload(StrictSchema):
    action_id = _id()
    rule_id = _id()
    arm = _one_of(*ARMS)
    channel = _one_of(*CHANNELS)


class ActionOutcomePayload(StrictSchema):
    action_id = _id()
    result = _one_of(*ACTION_RESULTS)


# --- Registry: the single list of event types ---

PAYLOAD_SCHEMAS = {
    "order_placed": OrderPlacedPayload,
    "order_delivered": OrderDeliveredPayload,
    "order_issue": OrderIssuePayload,
    "rating_submitted": RatingSubmittedPayload,
    "app_session": AppSessionPayload,
    "recommendation_shown": RecommendationShownPayload,
    "recommendation_clicked": RecommendationClickedPayload,
    "support_case": SupportCasePayload,
    "survey_response": SurveyResponsePayload,
    "action_sent": ActionSentPayload,
    "action_outcome": ActionOutcomePayload,
}

EVENT_TYPES = tuple(PAYLOAD_SCHEMAS)


# --- Envelope: the fields every event carries ---


class EnvelopeSchema(StrictSchema):
    """Fields every event carries, whatever its type."""

    event_id = fields.UUID(required=True)
    type = fields.String(required=True, validate=validate.OneOf(EVENT_TYPES))
    customer_id = _id()
    occurred_at = UtcDateTime(required=True)
    schema_version = fields.Integer(
        required=True, strict=True, validate=validate.Equal(SCHEMA_VERSION)
    )
    payload = fields.Dict(required=True)


class EventSchema(EnvelopeSchema):
    """A whole event: the envelope, then the payload checked against its type.

    Payload errors are nested under "payload". The payload is checked only
    once the envelope is valid, because its schema depends on the type.
    """

    @post_load
    def load_payload(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        payload_schema = PAYLOAD_SCHEMAS[data["type"]]()
        try:
            data["payload"] = payload_schema.load(data["payload"])
        except ValidationError as err:
            raise ValidationError(err.messages, field_name="payload") from err
        return data


def load_event(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one raw event and return it with its payload validated.

    Raises ValidationError. Envelope errors are keyed by field name;
    payload errors are nested under "payload".
    """
    event: dict[str, Any] = EventSchema().load(raw)
    return event
