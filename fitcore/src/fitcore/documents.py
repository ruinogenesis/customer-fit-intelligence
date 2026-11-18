"""Firestore documents v1: the shape of everything the system keeps (T301).

COLLECTIONS lists the nine collections, each with the schema its documents
must pass. check() validates one document before it is written: the API's
stores call it on every write, so a document of the wrong shape fails the
request instead of reaching Firestore. Reads are not checked.

The descriptions here are the documentation. fitcore/DOCUMENTS.md is made
from them (python -m fitcore.documents > fitcore/DOCUMENTS.md), and a test
fails if it is out of date.

Conventions, as for events (fitcore.schemas): ids are usable as document
ids; timestamps carry a timezone (Firestore keeps them in UTC); money is
whole satang. Scores are between 0 and 1.

Fields that later tasks add are not here yet; each task adds its own:
consent flags and the holdout bucket on customers (T610, T503), and an
expiry time on events (T305).

customer_document and order_document (and their inverses) convert
fitcore.scoring's Customer and Order, for the API and for Cloud Functions.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from marshmallow import Schema, ValidationError, fields, validate, validates_schema

from fitcore.expectations import Baseline
from fitcore.features import INPUTS
from fitcore.model import LATENTS, QUADRANTS, SCORES
from fitcore.schemas import (
    ACTION_RESULTS,
    ARMS,
    CHANNELS,
    ISSUE_TYPES,
    RATINGS,
    EventSchema,
    StrictBoolean,
    StrictSchema,
    check_id,
)
from fitcore.scoring import DRIVERS, PRIORS, Customer, Issue, Order, Scores

ROLES: tuple[str, ...] = ("admin", "analyst", "viewer")
MODEL_STATUSES: tuple[str, ...] = ("candidate", "live", "retired")
RULE_KINDS: tuple[str, ...] = ("service", "marketing")
RULE_STATUSES: tuple[str, ...] = ("dry_run", "test", "live", "off")


# --- Fields ---


def _id(description: str, *, allow_none: bool = False) -> fields.String:
    return fields.String(
        required=True,
        allow_none=allow_none,
        validate=[validate.Length(min=1), check_id],
        metadata={"description": description},
    )


def _text(description: str) -> fields.String:
    return fields.String(
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": description},
    )


def _choice(
    choices: tuple[str, ...], description: str, *, allow_none: bool = False
) -> fields.String:
    return fields.String(
        required=True,
        allow_none=allow_none,
        validate=validate.OneOf(choices),
        metadata={"description": description},
    )


def _time(description: str, *, allow_none: bool = False) -> fields.AwareDateTime:
    return fields.AwareDateTime(
        required=True, allow_none=allow_none, metadata={"description": description}
    )


def _satang(description: str) -> fields.Integer:
    return fields.Integer(
        required=True,
        strict=True,
        validate=validate.Range(min=0),
        metadata={"description": description},
    )


def _unit(description: str, *, allow_none: bool = False) -> fields.Float:
    return fields.Float(
        required=True,
        allow_none=allow_none,
        validate=validate.Range(min=0, max=1),
        metadata={"description": description},
    )


def _share(description: str) -> fields.Float:
    """A contribution to a score: between -1 and 1, as scores are 0 to 1."""
    return fields.Float(
        required=True,
        validate=validate.Range(min=-1, max=1),
        metadata={"description": description},
    )


def _distinct(names: list[str]) -> None:
    if len(set(names)) != len(names):
        raise ValidationError("Each input at most once.")


def _drivers(description: str, *, allow_none: bool = False) -> fields.List:
    return fields.List(
        fields.String(validate=validate.OneOf(INPUTS)),
        required=True,
        allow_none=allow_none,
        validate=[validate.Length(max=DRIVERS), _distinct],
        metadata={"description": description},
    )


def _flag(description: str) -> StrictBoolean:
    return StrictBoolean(required=True, metadata={"description": description})


def _map(description: str) -> fields.Dict:
    return fields.Dict(required=True, metadata={"description": description})


def _nested(schema: type[Schema], description: str, **kwargs: Any) -> fields.Nested:
    return fields.Nested(
        schema, required=True, metadata={"description": description}, **kwargs
    )


def _gs_path(description: str) -> fields.String:
    return fields.String(
        required=True,
        validate=validate.Regexp(r"^gs://[^/]+/.+"),
        metadata={"description": description},
    )


# --- Parts shared by several documents ---


class LatentsSchema(StrictSchema):
    """The five latents of one scoring (fitcore.model.LATENTS)."""

    respect = _unit("Respect")
    love = _unit("Love")
    convenience = _unit("Convenience value")
    trust = _unit("Trust")
    fit = _unit("Overall fit")


class ScoresSchema(LatentsSchema):
    """One order's scores, as POST /events answers them."""

    churn_risk = _unit("Chance of no order in the next 30 days")
    quadrant = _choice(QUADRANTS, "Love-matrix quadrant, from the model's lines (D5)")


class InputSharesSchema(StrictSchema):
    """Each input's contribution to one score (fitcore.model.attributions)."""

    x1 = _share("Speed vs promise")
    x2 = _share("App ease")
    x3 = _share("Fee load")
    x4 = _share("Order fidelity")
    x5 = _share("Courier conduct")
    x6 = _share("Food and selection")
    x7 = _share("Personalization and delight")
    x8 = _share("Recovery handling")


def _covers_every_score(value: Mapping[str, Any]) -> None:
    if set(value) != set(SCORES):
        raise ValidationError(f"Must cover exactly {', '.join(SCORES)}.")


class EventResultSchema(StrictSchema):
    """What POST /events answered for an event."""

    event_id = fields.UUID(
        required=True, metadata={"description": "The event's id, as text"}
    )
    customer_id = _id("The event's customer")
    duplicate = _flag("True if this event_id was already received")
    order_id = _id("The order this event re-scored; null if none", allow_none=True)
    scores = _nested(
        ScoresSchema, "That order's new scores; null if none changed", allow_none=True
    )
    drivers = _drivers(
        f"The inputs that moved that order's fit most, biggest first (at most "
        f"{DRIVERS}); null if no score changed",
        allow_none=True,
    )
    model_version = _text("The model that scored it")


class ExpectationSchema(StrictSchema):
    """One input's expectation (fitcore.expectations.Baseline)."""

    mean = fields.Float(
        required=True,
        metadata={"description": "Exponentially weighted mean, in the input's units"},
    )
    weight = fields.Float(
        required=True,
        validate=validate.Range(min=0),
        metadata={"description": "Effective number of observations behind it"},
    )
    last_seen = _time(
        "When the last observation came in; null before the first", allow_none=True
    )


class IssueSchema(StrictSchema):
    """One order_issue on an order."""

    issue_type = _choice(ISSUE_TYPES, "What went wrong")
    at = _time("When it was reported")


def _covers_every_prior(value: Mapping[str, Any]) -> None:
    if set(value) != set(PRIORS):
        raise ValidationError(f"Must cover exactly {', '.join(PRIORS)}.")


# --- The nine collections ---


class EventDocument(StrictSchema):
    """events/{event_id}: an event as received, and what it got back."""

    event = _nested(
        EventSchema, "The event as received and validated (fitcore.schemas)"
    )
    result = _nested(EventResultSchema, "What POST /events answered")


class CustomerDocument(StrictSchema):
    """customers/{customer_id}: a customer's expectations and latest scores."""

    customer_id = _id("The customer")
    expectations = fields.Dict(
        required=True,
        keys=fields.String(validate=validate.OneOf(tuple(PRIORS))),
        values=fields.Nested(ExpectationSchema),
        validate=_covers_every_prior,
        metadata={
            "description": "Expectation per input, x2..x8 (x1 is judged against "
            "the promise)",
            "keys": "x2..x8",
        },
    )
    latest_order_id = _id(
        "The latest order placed; null before the first", allow_none=True
    )
    latest_placed_at = _time(
        "When it was placed; null before the first order", allow_none=True
    )
    scores = _nested(
        LatentsSchema,
        "The latest order's latents; null before the first scoring",
        allow_none=True,
    )
    churn_risk = _unit(
        "The latest order's churn risk; null before the first", allow_none=True
    )
    quadrant = _choice(
        QUADRANTS,
        "The latest order's quadrant; null before the first",
        allow_none=True,
    )
    model_version = _text("The model behind the scores")

    @validates_schema
    def parts_go_together(self, data: dict[str, Any], **kwargs: Any) -> None:
        if (data["latest_order_id"] is None) != (data["latest_placed_at"] is None):
            raise ValidationError("latest_order_id and latest_placed_at go together.")
        scored = [data[name] is None for name in ("scores", "churn_risk", "quadrant")]
        if len(set(scored)) > 1:
            raise ValidationError("scores, churn_risk and quadrant go together.")


class ScoreEntryDocument(StrictSchema):
    """customers/{customer_id}/scores/{event_id}: one scoring of one order."""

    event_id = fields.UUID(
        required=True, metadata={"description": "The event that scored it, as text"}
    )
    order_id = _id("The order scored")
    at = _time("When that event occurred")
    scores = _nested(LatentsSchema, "The order's latents")
    churn_risk = _unit("The order's churn risk")
    quadrant = _choice(QUADRANTS, "The order's quadrant")
    contributions = fields.Dict(
        required=True,
        keys=fields.String(validate=validate.OneOf(SCORES)),
        values=fields.Nested(InputSharesSchema),
        validate=_covers_every_score,
        metadata={
            "description": "Each input's share of how far each score sits from an "
            "order exactly as expected (every gap 0); a score's eight add up to that "
            "distance (T115)",
            "keys": ", ".join(SCORES),
        },
    )
    drivers = _drivers(
        f"The inputs that moved fit most, up or down, biggest first (at most {DRIVERS})"
    )
    model_version = _text("The model that scored it")


class OrderDocument(StrictSchema):
    """orders/{order_id}: an order's record and its gaps so far."""

    order_id = _id("The order")
    customer_id = _id("Who placed it")
    restaurant_id = _id("Where from")
    placed_at = _time("When it was placed")
    promised_eta = _time("When it was promised")
    basket_value = _satang("Basket value, in satang")
    fees = _satang("Fees, in satang")
    promo_used = _flag("True if it used a promo")
    courier_id = _id("Who delivered it; null until delivered", allow_none=True)
    delivered_at = _time("When it was delivered; null until then", allow_none=True)
    issues = fields.List(
        fields.Nested(IssueSchema),
        required=True,
        metadata={"description": "Every order_issue, in the order received"},
    )
    ratings = fields.Dict(
        required=True,
        keys=fields.String(validate=validate.OneOf(RATINGS)),
        values=fields.Integer(strict=True, validate=validate.Range(min=1, max=5)),
        metadata={
            "description": "The latest rating of each kind, 1 to 5",
            "keys": ", ".join(RATINGS),
        },
    )
    gaps = fields.Dict(
        required=True,
        keys=fields.String(validate=validate.OneOf(INPUTS)),
        values=fields.Float(),
        metadata={
            "description": "The inputs this order has measured, in model units",
            "keys": "x1..x8",
        },
    )

    @validates_schema
    def delivery_goes_together(self, data: dict[str, Any], **kwargs: Any) -> None:
        if (data["courier_id"] is None) != (data["delivered_at"] is None):
            raise ValidationError("courier_id and delivered_at go together.")


class RuleDocument(StrictSchema):
    """rules/{rule_id}: an automation rule (T501 sets condition and action)."""

    rule_id = _id("The rule")
    name = _text("What staff call it")
    kind = _choice(
        RULE_KINDS, "service rules skip the marketing consent and frequency cap"
    )
    status = _choice(
        RULE_STATUSES, "dry_run logs only; test and live act; off is the kill switch"
    )
    holdout_share = _unit("Share of customers held out: 0.5 in test, 0.1 when live")
    condition = _map("When it fires; its form is set by T501")
    action = _map("What it sends; its form is set by T501")
    caps = _map("Its own limits; their form is set by T502")
    updated_at = _time("When it last changed")


class ActionDocument(StrictSchema):
    """actions/{action_id}: one decision of one rule, holdouts included."""

    action_id = _id("The action")
    rule_id = _id("The rule that decided it")
    customer_id = _id("For whom")
    arm = _choice(ARMS, "treatment acts; holdout is only logged")
    channel = _choice(CHANNELS, "Where it goes")
    dry_run = _flag("True if the rule was in dry_run, so nothing was sent")
    decided_at = _time("When the rule decided")
    outcome = _choice(
        ACTION_RESULTS, "The furthest it got; null until known", allow_none=True
    )


class ModelDocument(StrictSchema):
    """models/{version}: a model version in the registry (T310)."""

    version = _id("The model version")
    status = _choice(MODEL_STATUSES, "Only one version is live at a time")
    path = _gs_path("Its model.json in Cloud Storage; model_card.md sits next to it")
    sha256 = fields.String(
        required=True,
        validate=validate.Regexp(r"^[0-9a-f]{64}$"),
        metadata={"description": "model.json's SHA-256, checked before it is used"},
    )
    metrics = fields.Dict(
        required=True,
        keys=fields.String(validate=validate.Length(min=1)),
        values=fields.Float(),
        metadata={"description": "Held-out metrics by name, such as churn_auc"},
    )
    created_at = _time("When it was trained")
    promoted_at = _time("When it last went live; null if never", allow_none=True)


class UserDocument(StrictSchema):
    """users/{uid}: a dashboard user (T303)."""

    uid = _id("Their Firebase Auth uid")
    email = fields.Email(required=True, metadata={"description": "Their sign-in email"})
    role = _choice(ROLES, "Their role; T303 keeps it in step with the custom claim")
    updated_at = _time("When their role last changed")


# Written in the event's own transaction, and deleted once Pub/Sub has
# every message (cfi_api.export).
class OutboxDocument(StrictSchema):
    """outbox/{event_id}: an event's changes on their way to BigQuery (T306)."""

    event_id = fields.UUID(
        required=True, metadata={"description": "The event that wrote them, as text"}
    )
    messages = fields.List(
        fields.String(validate=validate.Length(min=1)),
        required=True,
        validate=validate.Length(min=1),
        metadata={
            "description": "One Pub/Sub message per document the event wrote: "
            "a row of raw.changes, as JSON text"
        },
    )
    created_at = _time("When the API wrote them: each message's written_at")


# --- The registry ---


@dataclass(frozen=True)
class Collection:
    """One collection: where its documents live, their schema, their writer."""

    path: str
    schema: type[Schema]
    one_per: str
    written_by: str


COLLECTIONS: Mapping[str, Collection] = {
    "events": Collection(
        "events/{event_id}", EventDocument, "event received", "the API"
    ),
    "customers": Collection(
        "customers/{customer_id}", CustomerDocument, "customer", "the API"
    ),
    "scores": Collection(
        "customers/{customer_id}/scores/{event_id}",
        ScoreEntryDocument,
        "scoring of an order",
        "the API",
    ),
    "orders": Collection("orders/{order_id}", OrderDocument, "order", "the API"),
    "rules": Collection(
        "rules/{rule_id}", RuleDocument, "automation rule", "the API, for admins"
    ),
    "actions": Collection(
        "actions/{action_id}", ActionDocument, "rule decision", "Cloud Functions"
    ),
    "models": Collection(
        "models/{version}", ModelDocument, "model version", "the training pipeline"
    ),
    "users": Collection("users/{uid}", UserDocument, "dashboard user", "admins"),
    "outbox": Collection(
        "outbox/{event_id}",
        OutboxDocument,
        "event whose changes Pub/Sub does not have yet",
        "the API",
    ),
}


# One schema object per collection, made once: making one costs more than
# checking a document with it.
_CHECKERS: Mapping[str, Schema] = {
    name: collection.schema() for name, collection in COLLECTIONS.items()
}


class DocumentError(ValueError):
    """A document that does not fit its collection's schema."""


def check(collection: str, document: Mapping[str, Any]) -> None:
    """Raise DocumentError unless document fits collection's schema."""
    errors = _CHECKERS[collection].validate(document)
    if errors:
        raise DocumentError(f"Not a valid {collection} document: {errors}")


# --- Customers and orders, to documents and back ---


def _latents(scores: Scores) -> dict[str, float]:
    return {name: getattr(scores, name) for name in LATENTS}


def customer_document(customer: Customer, model_version: str) -> dict[str, Any]:
    scores = customer.scores
    return {
        "customer_id": customer.customer_id,
        "expectations": {
            key: {"mean": b.mean, "weight": b.weight, "last_seen": b.last_seen}
            for key, b in customer.baselines.items()
        },
        "latest_order_id": customer.latest_order_id,
        "latest_placed_at": customer.latest_placed_at,
        "scores": None if scores is None else _latents(scores),
        "churn_risk": None if scores is None else scores.churn_risk,
        "quadrant": None if scores is None else scores.quadrant,
        "model_version": model_version,
    }


def customer_from_document(document: Mapping[str, Any]) -> Customer:
    scores = None
    if document["scores"] is not None:
        scores = Scores(
            **document["scores"],
            churn_risk=document["churn_risk"],
            quadrant=document["quadrant"],
        )
    return Customer(
        customer_id=document["customer_id"],
        baselines={
            key: Baseline(**value) for key, value in document["expectations"].items()
        },
        latest_order_id=document["latest_order_id"],
        latest_placed_at=document["latest_placed_at"],
        scores=scores,
    )


def order_document(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "restaurant_id": order.restaurant_id,
        "placed_at": order.placed_at,
        "promised_eta": order.promised_eta,
        "basket_value": order.basket_value,
        "fees": order.fees,
        "promo_used": order.promo_used,
        "courier_id": order.courier_id,
        "delivered_at": order.delivered_at,
        "issues": [{"issue_type": i.issue_type, "at": i.at} for i in order.issues],
        "ratings": dict(order.ratings),
        "gaps": dict(order.gaps),
    }


def order_from_document(document: Mapping[str, Any]) -> Order:
    return Order(
        document["order_id"],
        document["customer_id"],
        document["placed_at"],
        document["promised_eta"],
        document["gaps"],
        restaurant_id=document["restaurant_id"],
        basket_value=document["basket_value"],
        fees=document["fees"],
        promo_used=document["promo_used"],
        courier_id=document["courier_id"],
        delivered_at=document["delivered_at"],
        issues=tuple(Issue(**issue) for issue in document["issues"]),
        ratings=document["ratings"],
    )


# --- The reference, made from the schemas ---


def _kind(field: fields.Field) -> str:
    if check_id in field.validators:
        return "id"
    if isinstance(field, fields.UUID):
        return "UUID, as text"
    if isinstance(field, fields.DateTime):
        return "timestamp"
    if isinstance(field, fields.Boolean):
        return "boolean"
    if isinstance(field, fields.Integer):
        return "integer"
    if isinstance(field, fields.Float):
        return "number"
    if isinstance(field, fields.Email):
        return "email"
    if isinstance(field, fields.Nested | fields.Dict):
        return "map"
    if isinstance(field, fields.List) and isinstance(field.inner, fields.Nested):
        return "list of maps"
    if isinstance(field, fields.List):
        return "list of strings"
    return "string"


def _limits(field: fields.Field) -> str:
    notes = []
    for rule in field.validators:
        if isinstance(rule, validate.OneOf):
            notes.append("one of " + ", ".join(f"`{c}`" for c in rule.choices))
        elif isinstance(rule, validate.Range):
            notes.append(
                f"{rule.min:g} to {rule.max:g}"
                if rule.max is not None
                else f"{rule.min:g} or more"
            )
    if "keys" in field.metadata:
        notes.append(f"keys {field.metadata['keys']}")
    if field.allow_none:
        notes.append("or null")
    return "; ".join(notes)


def _rows(schema: Schema, prefix: str = "") -> list[str]:
    rows = []
    for name, field in schema.fields.items():
        kind, limits = _kind(field), _limits(field)
        text = field.metadata.get("description", "")
        rows.append(f"| `{prefix}{name}` | {kind} | {limits} | {text} |")
        inner: Schema | None = None
        path = f"{prefix}{name}."
        if isinstance(field, fields.Nested) and field.nested is not EventSchema:
            inner = field.schema
        elif isinstance(field, fields.Dict) and isinstance(
            field.value_field, fields.Nested
        ):
            inner, path = field.value_field.schema, f"{prefix}{name}.{{key}}."
        elif isinstance(field, fields.List) and isinstance(field.inner, fields.Nested):
            inner, path = field.inner.schema, f"{prefix}{name}[]."
        if inner is not None:
            rows += _rows(inner, path)
    return rows


def reference() -> str:
    """fitcore/DOCUMENTS.md: every collection and field, as Markdown."""
    lines = [
        "# Firestore documents",
        "",
        "Made from `fitcore.documents` (T301) by",
        "`python -m fitcore.documents > fitcore/DOCUMENTS.md`; do not edit by hand.",
        "The API checks every document against these schemas before it writes it.",
        "Timestamps carry a timezone and are kept in UTC; money is whole satang.",
    ]
    for collection in COLLECTIONS.values():
        doc = re.sub(r"^\S+: ", "", collection.schema.__doc__ or "")
        lines += [
            "",
            f"## `{collection.path}`",
            "",
            f"One per {collection.one_per}, written by {collection.written_by}: {doc}",
            "",
            "| Field | Type | Values | Meaning |",
            "| --- | --- | --- | --- |",
            *_rows(collection.schema()),
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(reference(), end="")
