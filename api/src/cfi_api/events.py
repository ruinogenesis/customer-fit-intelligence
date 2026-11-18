"""POST /api/v1/events: validate an event, store it once, re-score its customer.

receive() holds every step but the save, so POST /score (cfi_api.score)
answers exactly as this endpoint would, from a Reader that cannot save.
POST /events runs it through the store's handle(), which saves what it
changed in the same transaction as its reads (T302), then sends what was
saved on to BigQuery (cfi_api.export, T306).
"""

from datetime import UTC, datetime
from typing import Any

from fitcore.documents import EventResultSchema
from fitcore.schemas import EventSchema
from fitcore.scoring import Applied, Customer, EventConflict, apply_event, order_id_for
from flask import abort
from flask.views import MethodView
from flask_smorest import Blueprint
from flask_smorest import abort as abort_with

from cfi_api.access import ACCESS
from cfi_api.errors import ErrorSchema
from cfi_api.export import export
from cfi_api.model_registry import BadModel
from cfi_api.runtime import live_model, publisher, store
from cfi_api.store import Busy, Reader

blp = Blueprint("events", __name__, description="Events in, scores out")

# Seconds a sender should wait before sending an event the store was too busy for.
RETRY_AFTER = 1


def as_stored(event: dict[str, Any]) -> dict[str, Any]:
    """A validated event as it is stored and compared: event_id as text."""
    return {**event, "event_id": str(event["event_id"])}


def receive(event: dict[str, Any], db: Reader) -> tuple[dict[str, Any], Applied | None]:
    """What POST /events answers for event (as_stored), and what it changes.

    The change is None when this event_id was already received: the answer
    is then its first result, with duplicate set to true. Only reads db.
    Aborts with 503 without a model, and 409 for an event that does not fit
    what is stored or reuses an event_id for a different event.
    """
    try:
        model = live_model(db)
    except BadModel as err:
        abort(503, description=f"The live model cannot be loaded: {err}")
    if model is None:
        abort(503, description="No model is loaded, so events cannot be scored.")
    stored = db.find_event(event["event_id"])
    if stored is not None:
        if stored.event != event:
            abort(409, description="This event_id was used for a different event.")
        return {**stored.result, "duplicate": True}, None

    customer_id = event["customer_id"]
    customer = db.find_customer(customer_id) or Customer.new(customer_id)
    order_id = order_id_for(event, customer)
    order = None if order_id is None else db.find_order(order_id)
    try:
        applied = apply_event(model.params, event, customer, order)
    except EventConflict as err:
        abort(409, description=str(err))
    scored, explained = applied.scored, applied.explained
    result = {
        "event_id": event["event_id"],
        "customer_id": customer_id,
        "duplicate": False,
        "order_id": applied.order.order_id if scored and applied.order else None,
        "scores": None if scored is None else scored.to_dict(),
        "drivers": None if explained is None else list(explained.drivers),
        "model_version": model.version,
    }
    return result, applied


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
    @blp.alt_response(
        401, schema=ErrorSchema, description="No valid sign-in token or X-CFI-Key"
    )
    @blp.alt_response(403, schema=ErrorSchema, description="Only for admins")
    @blp.alt_response(
        503,
        schema=ErrorSchema,
        description="No model is loaded, or too many events for one customer at "
        "once: send it again after Retry-After seconds",
    )
    @blp.doc(security=ACCESS["events.Events", "POST"].security)
    def post(self, event: dict[str, Any]) -> Any:
        """Ingest one event and re-score its customer.

        Idempotent by event_id for 30 days: sending the same event again
        within 30 days of the first returns its first result, with duplicate
        set to true, and changes nothing. After that the stored event has
        expired and a repeat counts as new.
        """
        event = as_stored(event)
        try:
            result, applied, entry = store().handle(
                event, lambda db: receive(event, db)
            )
        except Busy:
            abort_with(
                503,
                message="Too many events for this customer at once; "
                "send it again shortly.",
                headers={"Retry-After": str(RETRY_AFTER)},
            )
        to_bigquery = publisher()
        if to_bigquery is not None:
            export(store(), to_bigquery, entry, datetime.now(UTC))
        if applied is None:
            return result, 200
        return result
