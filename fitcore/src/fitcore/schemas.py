"""Event schemas v1: the contract for every event that enters the system."""

from datetime import UTC

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


def _id(required=True):
    return fields.String(required=required, validate=validate.Length(min=1))


def _count():
    """A whole number of 0 or more: money in satang, counts, positions."""
    return fields.Integer(required=True, strict=True, validate=validate.Range(min=0))


def _score(low, high, required=True):
    return fields.Integer(
        required=required, strict=True, validate=validate.Range(min=low, max=high)
    )


def _one_of(*choices):
    return fields.String(required=True, validate=validate.OneOf(choices))


class StrictSchema(Schema):
    """Base for every schema in this module: unknown fields are rejected."""

    class Meta:
        unknown = RAISE


# --- Payloads: one schema per event type ---


class OrderPlacedPayload(StrictSchema):
    order_id = _id()
    restaurant_id = _id()
    basket_value = _count()
    fees = _count()
    promised_eta = fields.AwareDateTime(required=True)
    promo_used = fields.Boolean(required=True, truthy={True}, falsy={False})


class OrderDeliveredPayload(StrictSchema):
    order_id = _id()
    courier_id = _id()


class OrderIssuePayload(StrictSchema):
    order_id = _id()
    issue_type = _one_of("missing", "wrong", "cold", "late")


class RatingSubmittedPayload(StrictSchema):
    order_id = _id()
    food = _score(1, 5, required=False)
    courier = _score(1, 5, required=False)
    accuracy = _score(1, 5, required=False)
    overall = _score(1, 5, required=False)

    @validates_schema
    def at_least_one_score(self, data, **kwargs):
        if not any(name in data for name in ("food", "courier", "accuracy", "overall")):
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
    opened_at = fields.AwareDateTime(required=True)
    refund_amount = _count()
    satisfaction = _score(1, 5, required=False)


class SurveyResponsePayload(StrictSchema):
    item_id = _id()
    value = _score(1, 7)


class ActionSentPayload(StrictSchema):
    action_id = _id()
    rule_id = _id()
    arm = _one_of("treatment", "holdout")
    channel = _one_of("push", "in_app", "email", "voucher")


class ActionOutcomePayload(StrictSchema):
    action_id = _id()
    result = _one_of("delivered", "opened", "clicked", "redeemed", "failed")


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
    occurred_at = fields.AwareDateTime(required=True)
    schema_version = fields.Integer(
        required=True, strict=True, validate=validate.Equal(SCHEMA_VERSION)
    )
    payload = fields.Dict(required=True)

    @post_load
    def convert_to_utc(self, data, **kwargs):
        data["occurred_at"] = data["occurred_at"].astimezone(UTC)
        return data

def load_event(raw):
    """Validate one raw event and return it with its payload validated.

    Raises ValidationError. Envelope errors are keyed by field name;
    payload errors are nested under "payload".
    """
    event = EnvelopeSchema().load(raw)
    payload_schema = PAYLOAD_SCHEMAS[event["type"]]()
    try:
        event["payload"] = payload_schema.load(event["payload"])
    except ValidationError as err:
        raise ValidationError({"payload": err.messages}) from err
    return event
