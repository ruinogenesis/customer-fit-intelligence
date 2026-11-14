"""POST /api/v1/events: validate an event, store it once, re-score its customer."""

from typing import Any

from fitcore.schemas import EventSchema
from fitcore.scoring import Customer, EventConflict, apply_event, order_id_for
from flask import abort
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import Schema, fields

from cfi_api.errors import ErrorSchema
from cfi_api.runtime import live_model, store
from cfi_api.store import EventRecord

blp = Blueprint("events", __name__, description="Events in, scores out")


class ScoresSchema(Schema):
    respect = fields.Float(required=True)
    love = fields.Float(required=True)
    convenience = fields.Float(required=True)
    trust = fields.Float(required=True)
    fit = fields.Float(required=True)
    churn_risk = fields.Float(required=True)


class EventResultSchema(Schema):
    event_id = fields.String(required=True)
    customer_id = fields.String(required=True)
    duplicate = fields.Boolean(
        required=True,
        metadata={"description": "True if this event_id was already received"},
    )
    order_id = fields.String(
        required=True,
        allow_none=True,
        metadata={"description": "The order this event re-scored; null if none"},
    )
    scores = fields.Nested(
        ScoresSchema,
        required=True,
        allow_none=True,
        metadata={"description": "That order's new scores; null if none changed"},
    )
    model_version = fields.String(required=True)


@blp.route("/events")
class Events(MethodView):
    @blp.arguments(EventSchema)
    @blp.response(201, EventResultSchema)
    @blp.alt_response(
        200, schema=EventResultSchema, description="Already received: the first result"
    )
    @blp.alt_response(
        409, schema=ErrorSchema, description="Valid, but does not fit what is stored"
    )
    @blp.alt_response(503, schema=ErrorSchema, description="No model is loaded")
    def post(self, event: dict[str, Any]) -> Any:
        """Ingest one event and re-score its customer.

        Idempotent by event_id: sending the same event again returns its
        first result, with duplicate set to true, and changes nothing.
        """
        model = live_model()
        if model is None:
            abort(503, description="No model is loaded, so events cannot be scored.")
        event = {**event, "event_id": str(event["event_id"])}
        db = store()
        stored = db.find_event(event["event_id"])
        if stored is not None:
            if stored.event != event:
                abort(409, description="This event_id was used for a different event.")
            return {**stored.result, "duplicate": True}, 200

        customer_id = event["customer_id"]
        customer = db.find_customer(customer_id) or Customer.new(customer_id)
        order_id = order_id_for(event, customer)
        order = None if order_id is None else db.find_order(order_id)
        try:
            applied = apply_event(model.params, event, customer, order)
        except EventConflict as err:
            abort(409, description=str(err))
        scored = applied.scored
        result = {
            "event_id": event["event_id"],
            "customer_id": customer_id,
            "duplicate": False,
            "order_id": applied.order.order_id if scored and applied.order else None,
            "scores": None if scored is None else scored.to_dict(),
            "model_version": model.version,
        }
        db.save(EventRecord(event, result), applied, model.version)
        return result
