"""Scoring from events: what each event measures, and the scores it changes.

The API and the notebooks share this module, so an event log replayed
through either gives the same scores (the golden test, T309). It stores
nothing: the caller loads the customer and the order an event is about
(order_id_for says which), calls apply_event, and saves what comes back.

What each event measures, in the units the model was trained on:

- order_placed opens the order with its promised ETA, and measures x3, fees
  divided by basket value (nothing when the basket is 0).
- order_delivered measures x1: the promised minutes minus the actual ones.
- app_session measures x2, seconds to checkout (null: no checkout), for the
  customer's latest order, if it was placed before the session ended.
- rating_submitted measures x5 from the courier score and x6 from the food
  score.
- support_case measures x8 from satisfaction (1 to 5), for its order or
  else the customer's latest.

x4 and x7 have no measure in the training units yet. order_issue and the
other event types measure nothing, so they change no score.

consent_changed (T610) measures nothing either: it sets the customer's
marketing consent, unless the customer already holds a later one, so an
event that arrives late never undoes a newer choice.

Each order also keeps its record (T301): restaurant, basket, fees and promo
from order_placed; courier and delivery time from order_delivered; every
order_issue; and the latest score of each kind from rating_submitted.

x2..x8 are judged against the customer's own expectation
(fitcore.expectations), read just before the observation updates it, as the
simulator does; x1 is judged against the promise. Each order is scored on
what was measured for that order alone: an input it has not measured counts
as 0, "as expected" (T202: this beat carrying the last gap forward). The
customer's scores are those of their latest order.

Every scoring is also explained (T316): explain() splits each score's
distance from an order exactly as expected across the eight inputs
(fitcore.model.attributions, T115), and names fit's top drivers.
"""

from collections.abc import Mapping
from dataclasses import KW_ONLY, asdict, dataclass, field, replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any

import numpy as np

from fitcore import expectations
from fitcore.features import INPUTS, input_gap
from fitcore.model import (
    QUADRANTS,
    SCORES,
    ModelParams,
    attributions,
    forward,
    outcomes,
    quadrant,
)
from fitcore.schemas import RATINGS

# The population's usual value of x2..x8: a new customer's expectation, in
# the model's units. Today these are the simulator's (fitcore.sim_orders);
# real data will set them along with the model.
PRIORS: Mapping[str, float] = MappingProxyType(
    {
        "x2": 90.0,  # seconds to checkout
        "x3": 0.12,  # fee-to-basket ratio
        "x4": 0.97,  # share of items right
        "x5": 4.5,  # courier rating, 1 to 5
        "x6": 4.3,  # food rating, 1 to 5
        "x7": 0.3,  # recommendation hit rate
        "x8": 4.0,  # recovery score, 1 to 5
    }
)

# How many of fit's drivers each scoring names (T316).
DRIVERS = 3

# Event types that always name their order: order_placed opens it, and the
# others need it to exist already.
ORDER_EVENTS: tuple[str, ...] = (
    "order_placed",
    "order_delivered",
    "order_issue",
    "rating_submitted",
)


class EventConflict(ValueError):
    """A valid event that does not fit what is already stored.

    Its message is a sentence for people, ready for an API error.
    """


@dataclass(frozen=True)
class Scores:
    """One order's five latents (fitcore.model.LATENTS order), its churn risk,
    and its love-matrix quadrant (one of fitcore.model.QUADRANTS)."""

    respect: float
    love: float
    convenience: float
    trust: float
    fit: float
    churn_risk: float
    quadrant: str

    def __post_init__(self) -> None:
        if self.quadrant not in QUADRANTS:
            raise ValueError(
                f"quadrant must be one of {QUADRANTS}, got {self.quadrant!r}"
            )

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class Explanation:
    """How one order's inputs moved its scores away from an order exactly as
    expected, every gap 0 (fitcore.model.attributions, T115).

    contributions: by score (fitcore.model.SCORES), then by input (INPUTS),
    all 48, zeros included; a score's eight add up to its change. drivers:
    the inputs that moved fit most, up or down, biggest first: at most
    DRIVERS, fewer when fewer inputs moved fit at all.
    """

    contributions: dict[str, dict[str, float]]
    drivers: tuple[str, ...]


@dataclass(frozen=True)
class Issue:
    """One order_issue: what went wrong, and when it was reported."""

    issue_type: str
    at: datetime


@dataclass(frozen=True)
class Order:
    """One order: who placed it, when, what was promised, its gaps so far,
    and its record.

    gaps holds only the inputs this order has measured (x1..x8, own units).
    The record: restaurant_id, basket_value and fees (satang) and promo_used
    from order_placed; courier_id and delivered_at once delivered; issues in
    the order reported; ratings, the latest score of each kind (RATINGS).
    """

    order_id: str
    customer_id: str
    placed_at: datetime
    promised_eta: datetime
    gaps: Mapping[str, float] = field(default_factory=dict)
    _: KW_ONLY
    restaurant_id: str
    basket_value: int
    fees: int
    promo_used: bool
    courier_id: str | None = None
    delivered_at: datetime | None = None
    issues: tuple[Issue, ...] = ()
    ratings: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.gaps) - set(INPUTS)
        if unknown:
            raise ValueError(f"unknown inputs in gaps: {sorted(unknown)}")
        unknown = set(self.ratings) - set(RATINGS)
        if unknown:
            raise ValueError(f"unknown ratings: {sorted(unknown)}")
        if (self.courier_id is None) != (self.delivered_at is None):
            raise ValueError("courier_id and delivered_at go together")
        object.__setattr__(self, "gaps", MappingProxyType(dict(self.gaps)))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "ratings", MappingProxyType(dict(self.ratings)))


@dataclass(frozen=True)
class Consent:
    """A customer's latest marketing consent (T610): given or withdrawn, when
    and where (fitcore.schemas.CONSENT_SOURCES)."""

    marketing: bool
    at: datetime
    source: str


@dataclass(frozen=True)
class Customer:
    """One customer's expectations (x2..x8), latest order and scores, and
    their marketing consent (None: never asked, which is no consent).

    Use Customer.new for a customer with no history yet.
    """

    customer_id: str
    baselines: Mapping[str, expectations.Baseline]
    latest_order_id: str | None = None
    latest_placed_at: datetime | None = None
    scores: Scores | None = None
    consent: Consent | None = None

    def __post_init__(self) -> None:
        if set(self.baselines) != set(PRIORS):
            raise ValueError(f"baselines must cover exactly {sorted(PRIORS)}")
        if (self.latest_order_id is None) != (self.latest_placed_at is None):
            raise ValueError("latest_order_id and latest_placed_at go together")
        object.__setattr__(self, "baselines", MappingProxyType(dict(self.baselines)))

    @classmethod
    def new(cls, customer_id: str) -> "Customer":
        return cls(customer_id, {key: expectations.Baseline() for key in PRIORS})


@dataclass(frozen=True)
class Applied:
    """What one event changed.

    customer: the customer afterwards (always saved). order: the order the
    event created or changed (its gaps or its record), None if none. scored:
    that order's new scores, None if no score changed. explained: how its
    inputs moved them (T316), exactly when scored is not None.
    """

    customer: Customer
    order: Order | None
    scored: Scores | None
    explained: Explanation | None = None

    def __post_init__(self) -> None:
        if (self.scored is None) != (self.explained is None):
            raise ValueError("scored and explained go together")


def measures(event: Mapping[str, Any]) -> dict[str, float]:
    """The raw values of x2..x8 that one validated event measures.

    x1 is not here: it needs the order's promise (see apply_event).
    """
    payload = event["payload"]
    match event["type"]:
        case "order_placed":
            basket = payload["basket_value"]
            return {"x3": payload["fees"] / basket} if basket > 0 else {}
        case "app_session":
            seconds = payload["time_to_checkout_s"]
            return {} if seconds is None else {"x2": float(seconds)}
        case "rating_submitted":
            names = {"x5": "courier", "x6": "food"}
            return {key: float(payload[n]) for key, n in names.items() if n in payload}
        case "support_case":
            if "satisfaction" not in payload:
                return {}
            return {"x8": float(payload["satisfaction"])}
        case _:
            return {}


def order_id_for(event: Mapping[str, Any], customer: Customer) -> str | None:
    """The order this event is about: the one to load before apply_event."""
    payload = event["payload"]
    kind = event["type"]
    if kind in ORDER_EVENTS:
        order_id: str = payload["order_id"]
        return order_id
    if kind == "support_case":
        return payload.get("order_id") or customer.latest_order_id
    if kind == "app_session" and payload["time_to_checkout_s"] is not None:
        return customer.latest_order_id
    return None


def _row(gaps: Mapping[str, float]) -> np.ndarray:
    """One order's gaps as a 1 x 8 row; an input missing from gaps is 0."""
    unknown = set(gaps) - set(INPUTS)
    if unknown:
        raise ValueError(f"unknown inputs in gaps: {sorted(unknown)}")
    return np.array([[gaps.get(key, 0.0) for key in INPUTS]], dtype=np.float64)


def score(params: ModelParams, gaps: Mapping[str, float]) -> Scores:
    """The scores for one order's gaps; an input missing from gaps counts as 0.

    The quadrant uses the model's own lines (D5).
    """
    latents = forward(params, _row(gaps))[0]
    churn = outcomes(params, latents[-1:]).churn[0]
    respect, love, convenience, trust, fit = (float(value) for value in latents)
    place = quadrant(params, respect, love)
    return Scores(respect, love, convenience, trust, fit, float(churn), place)


def explain(params: ModelParams, gaps: Mapping[str, float]) -> Explanation:
    """How one order's gaps moved its scores (T316); an input missing from
    gaps counts as 0, as score() counts it.

    Fit's drivers go by the size of their contribution, up or down; a tie
    keeps the inputs' order, and an input that did not move fit is left out.
    """
    split = attributions(params, _row(gaps))[0]
    contributions = {
        name: dict(zip(INPUTS, (float(value) for value in split[k]), strict=True))
        for k, name in enumerate(SCORES)
    }
    fit = contributions["fit"]
    moved = sorted(
        (key for key in INPUTS if fit[key] != 0.0), key=lambda k: -abs(fit[k])
    )
    return Explanation(contributions, tuple(moved[:DRIVERS]))


def _minutes(span: timedelta) -> float:
    return span.total_seconds() / 60.0


def _record(event: Mapping[str, Any], order: Order) -> dict[str, Any]:
    """What this event adds to its order's record, as Order fields to replace."""
    payload = event["payload"]
    match event["type"]:
        case "order_delivered":
            return {
                "courier_id": payload["courier_id"],
                "delivered_at": event["occurred_at"],
            }
        case "order_issue":
            issue = Issue(payload["issue_type"], event["occurred_at"])
            return {"issues": (*order.issues, issue)}
        case "rating_submitted":
            scores = {name: payload[name] for name in RATINGS if name in payload}
            return {"ratings": {**order.ratings, **scores}}
        case _:
            return {}


def apply_event(
    params: ModelParams,
    event: Mapping[str, Any],
    customer: Customer,
    order: Order | None,
) -> Applied:
    """Apply one validated event (fitcore.schemas.load_event) to its customer.

    order must be the stored order that order_id_for names, or None if there
    is none. Raises EventConflict when the event does not fit what is
    stored: an order placed twice, an order that has not been placed, an
    order of another customer, or a delivery before the order was placed.
    Raises ValueError if the caller passes the wrong customer or order.
    """
    kind = event["type"]
    payload = event["payload"]
    at: datetime = event["occurred_at"]
    if event["customer_id"] != customer.customer_id:
        raise ValueError("the event is about another customer")
    wanted = order_id_for(event, customer)
    if order is not None and order.order_id != wanted:
        raise ValueError(f"the event is about order {wanted!r}, not {order.order_id!r}")

    named = kind in ORDER_EVENTS or (kind == "support_case" and "order_id" in payload)
    if kind == "order_placed":
        if order is not None:
            raise EventConflict(f"Order {wanted!r} was already placed.")
        order = Order(
            payload["order_id"],
            customer.customer_id,
            at,
            payload["promised_eta"],
            restaurant_id=payload["restaurant_id"],
            basket_value=payload["basket_value"],
            fees=payload["fees"],
            promo_used=payload["promo_used"],
        )
    elif named and order is None:
        raise EventConflict(f"Order {wanted!r} has not been placed.")
    if order is not None and order.customer_id != customer.customer_id:
        raise EventConflict(f"Order {wanted!r} belongs to another customer.")
    if kind == "order_delivered" and order is not None and at < order.placed_at:
        raise EventConflict(f"Order {wanted!r} was delivered before it was placed.")
    if kind == "app_session" and order is not None and order.placed_at > at:
        order = None  # that order came after this session: not its checkout

    baselines = dict(customer.baselines)
    gaps: dict[str, float] = {}
    for key, value in measures(event).items():
        expected = expectations.expected(baselines[key], PRIORS[key], at)
        baselines[key] = expectations.update(baselines[key], value, at)
        gaps[key] = input_gap(key, value, expected)
    if kind == "order_delivered" and order is not None:
        actual = _minutes(at - order.placed_at)
        promised = _minutes(order.promised_eta - order.placed_at)
        gaps["x1"] = input_gap("x1", actual, promised)

    latest_id, latest_at = customer.latest_order_id, customer.latest_placed_at
    if kind == "order_placed" and (latest_at is None or at >= latest_at):
        latest_id, latest_at = payload["order_id"], at

    changed: Order | None = None
    scored: Scores | None = None
    explained: Explanation | None = None
    if order is not None:
        record = _record(event, order)
        if gaps or record or kind == "order_placed":
            changed = replace(order, gaps={**order.gaps, **gaps}, **record)
        if changed is not None and (gaps or kind == "order_placed"):
            scored = score(params, changed.gaps)
            explained = explain(params, changed.gaps)
    latest_scores = customer.scores
    if scored is not None and changed is not None and changed.order_id == latest_id:
        latest_scores = scored
    consent = customer.consent
    if kind == "consent_changed" and (consent is None or at >= consent.at):
        consent = Consent(payload["marketing"], at, payload["source"])
    return Applied(
        customer=replace(
            customer,
            baselines=baselines,
            latest_order_id=latest_id,
            latest_placed_at=latest_at,
            scores=latest_scores,
            consent=consent,
        ),
        order=changed,
        scored=scored,
        explained=explained,
    )
