"""Event schemas v1: the contract for every event that enters the system."""

from datetime import UTC

from marshmallow import RAISE, Schema, fields, post_load, validate

SCHEMA_VERSION = 1

EVENT_TYPES = (
    "order_placed",
    "order_delivered",
    "order_issue",
    "rating_submitted",
    "app_session",
    "recommendation_shown",
    "recommendation_clicked",
    "support_case",
    "survey_response",
    "action_sent",
    "action_outcome",
)


class EnvelopeSchema(Schema):
    """Fields every event carries, whatever its type."""

    class Meta:
        unknown = RAISE

    event_id = fields.UUID(required=True)
    type = fields.String(required=True, validate=validate.OneOf(EVENT_TYPES))
    customer_id = fields.String(required=True, validate=validate.Length(min=1))
    occurred_at = fields.AwareDateTime(required=True)
    schema_version = fields.Integer(
        required=True, strict=True, validate=validate.Equal(SCHEMA_VERSION)
    )
    payload = fields.Dict(required=True)

    @post_load
    def convert_to_utc(self, data, **kwargs):
        data["occurred_at"] = data["occurred_at"].astimezone(UTC)
        return data