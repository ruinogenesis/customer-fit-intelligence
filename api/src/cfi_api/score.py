"""POST /api/v1/score: what POST /events would answer for an event, saving nothing."""

from typing import Any

from fitcore.documents import EventResultSchema
from fitcore.schemas import EventSchema
from flask.views import MethodView
from flask_smorest import Blueprint

from cfi_api.errors import ErrorSchema
from cfi_api.events import as_stored, receive
from cfi_api.runtime import store
from cfi_api.write_key import SCHEME

blp = Blueprint("score", __name__, description="What-if scoring: nothing is saved")


@blp.route("/score")
class Score(MethodView):
    @blp.arguments(EventSchema)
    @blp.response(200, EventResultSchema)
    @blp.alt_response(
        409, schema=ErrorSchema, description="Valid, but does not fit what is stored"
    )
    @blp.alt_response(
        401,
        schema=ErrorSchema,
        description="No valid X-CFI-Key, on a server with a write key",
    )
    @blp.alt_response(503, schema=ErrorSchema, description="No model is loaded")
    @blp.doc(security=[{SCHEME: []}])
    def post(self, event: dict[str, Any]) -> Any:
        """Score a hypothetical event without saving anything.

        The answer is what POST /events would give for the same event right
        now, from the customer's stored expectations and order. Nothing is
        saved, so the same event can be scored any number of times.
        """
        result, _ = receive(as_stored(event), store())
        return result
