"""Simulated orders as v1 events: what fake customers' apps would send.

fitcore.sim_orders simulates what each order delivers. This module writes
those orders as the raw events the API takes (fitcore.schemas, v1), in time
order. Every order sends:

- order_placed when it is placed: fees over basket value give x3.
- app_session 30 seconds later: its time to checkout is x2.
- order_delivered when it arrives: minutes taken against the promise, x1.

Some orders send more, after the delivery:

- rating_submitted 10 minutes later, on about RATING_RATE of orders and on
  every order with an incident: the courier score is x5, the food score x6,
  in whole stars.
- On an order with an incident, order_issue 5 minutes later when the
  incident has an issue type (ISSUE_TYPES), and a support_case opened 15
  minutes later and closed 30 minutes after that: its satisfaction is x8.

With plant=True (T315), most incidents of some kinds go to a few named
restaurants and couriers (PLANTED), so a check can tell whether /drivers
finds them. Only who made or brought those orders changes: every gap, time
and event id stays as it is without planting.

The seed decides everything, so the same arguments always give the same
events, ids included. Ids carry the seed ("sim1-000003"), so runs with
different seeds never share a customer. A customer's orders, and the details
drawn for each order, do not depend on how many customers or days are
simulated: a smaller run's events are all part of a bigger run's.
"""

import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any

import numpy as np

from fitcore.features import INPUTS
from fitcore.schemas import SCHEMA_VERSION
from fitcore.sim_orders import DAYS, START, simulate_orders, simulate_profiles

# The share of orders without an incident that get a rating.
RATING_RATE = 0.3

# The issue type an incident is reported as; the others have none.
ISSUE_TYPES: Mapping[str, str] = MappingProxyType(
    {"late": "late", "wrong_items": "wrong", "cold_food": "cold"}
)

# Baskets in satang: lognormal around a median of 350 baht, at least 50.
BASKET_MEDIAN = 35_000
BASKET_SIGMA = 0.4
BASKET_LOW = 5_000

# Wrong items or cold food get this share of the basket refunded.
REFUND_SHARE = 0.2

# Each order is from one of RESTAURANTS restaurants and brought by one of
# COURIERS couriers; PROMO_RATE of orders use a promo.
RESTAURANTS = 40
COURIERS = 60
PROMO_RATE = 0.2

# Planted trouble (T315): with plant=True, PLANT_SHARE of each kind of
# incident below goes to one of a few restaurants or couriers, drawn from
# the order's own planting stream. The kitchens behind late and cold orders,
# and the couriers behind rude ones.
PLANTED: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "late": ("restaurant_id", ("rest-00", "rest-01")),
        "cold_food": ("restaurant_id", ("rest-02",)),
        "rude_courier": ("courier_id", ("courier-00", "courier-01", "courier-02")),
    }
)
PLANT_SHARE = 0.8

# Event ids are uuid5 names in this namespace, so they never change.
NAMESPACE = uuid.UUID("0b6f3f0e-7a1d-4c2e-9f5a-2d8c4e6b1a37")


def stars(value: float) -> int:
    """A rating as whole stars: the nearest of 1 to 5, halves rounded up."""
    return int(np.clip(np.floor(value + 0.5), 1, 5))


def _planted(
    incident: str | None, planting: np.random.Generator | None
) -> dict[str, str]:
    """Where a PLANTED incident moves its order: {field: name}, or nothing."""
    if planting is None or incident is None or incident not in PLANTED:
        return {}
    field, names = PLANTED[incident]
    if planting.random() >= PLANT_SHARE:
        return {}
    return {field: names[int(planting.integers(len(names)))]}


def _order_events(
    customer_id: str,
    order_id: str,
    placed: datetime,
    promise: float,
    measured: Mapping[str, float],
    incident: str | None,
    rng: np.random.Generator,
    planting: np.random.Generator | None = None,
) -> list[tuple[datetime, dict[str, Any]]]:
    """One order's raw events, each with its time for sorting.

    An order sends each event type at most once, so the order id and the
    type are enough to name an event: they make its event_id. With a
    planting stream, a PLANTED incident may move the order to one of its
    restaurants or couriers; rng draws exactly as it would without one.
    """
    events: list[tuple[datetime, dict[str, Any]]] = []

    def add(kind: str, at: datetime, payload: dict[str, Any]) -> None:
        event = {
            "event_id": str(uuid.uuid5(NAMESPACE, f"{order_id}/{kind}")),
            "type": kind,
            "customer_id": customer_id,
            "occurred_at": at.isoformat(),
            "schema_version": SCHEMA_VERSION,
            "payload": payload,
        }
        events.append((at, event))

    basket = max(BASKET_LOW, round(rng.lognormal(np.log(BASKET_MEDIAN), BASKET_SIGMA)))
    delivered = placed + timedelta(minutes=measured["x1"])
    restaurant = f"rest-{int(rng.integers(RESTAURANTS)):02d}"
    moved = _planted(incident, planting)
    add(
        "order_placed",
        placed,
        {
            "order_id": order_id,
            "restaurant_id": moved.get("restaurant_id", restaurant),
            "basket_value": basket,
            "fees": round(basket * measured["x3"]),
            "promised_eta": (placed + timedelta(minutes=promise)).isoformat(),
            "promo_used": bool(rng.random() < PROMO_RATE),
        },
    )
    add(
        "app_session",
        placed + timedelta(seconds=30),
        {
            "session_id": f"{order_id}-session",
            "time_to_checkout_s": round(measured["x2"]),
            "steps": int(rng.integers(3, 9)),
            "errors": 0,
        },
    )
    courier = f"courier-{int(rng.integers(COURIERS)):02d}"
    add(
        "order_delivered",
        delivered,
        {"order_id": order_id, "courier_id": moved.get("courier_id", courier)},
    )
    if incident is not None or rng.random() < RATING_RATE:
        add(
            "rating_submitted",
            delivered + timedelta(minutes=10),
            {
                "order_id": order_id,
                "courier": stars(measured["x5"]),
                "food": stars(measured["x6"]),
            },
        )
    if incident is not None:
        if incident in ISSUE_TYPES:
            add(
                "order_issue",
                delivered + timedelta(minutes=5),
                {"order_id": order_id, "issue_type": ISSUE_TYPES[incident]},
            )
        opened = delivered + timedelta(minutes=15)
        refunded = incident in ("wrong_items", "cold_food")
        add(
            "support_case",
            opened + timedelta(minutes=30),
            {
                "case_id": f"{order_id}-case",
                "order_id": order_id,
                "opened_at": opened.isoformat(),
                "refund_amount": round(basket * REFUND_SHARE) if refunded else 0,
                "satisfaction": stars(measured["x8"]),
            },
        )
    return events


def simulate_events(
    customers: int, seed: int, days: float = DAYS, *, plant: bool = False
) -> list[dict[str, Any]]:
    """The raw events of `customers` simulated customers over `days` days.

    The orders come from fitcore.sim_orders (simulate_profiles, then
    simulate_orders from its START), and the events are in time order. With
    plant, PLANTED restaurants and couriers take most of their incidents
    (T315). Raises ValueError if customers is below 1 or days is not positive.
    """
    profiles = simulate_profiles(customers, seed)
    orders = simulate_orders(profiles, seed, start=START, days=days)
    position = {profile.customer_id: i for i, profile in enumerate(profiles)}
    promise = {profile.customer_id: profile.promise for profile in profiles}
    timed: list[tuple[datetime, dict[str, Any]]] = []
    for row, sim_id in enumerate(orders.customer_ids):
        number = orders.order_numbers[row]
        customer_id = f"sim{seed}-{position[sim_id]:06d}"
        timed += _order_events(
            customer_id,
            order_id=f"{customer_id}-{number:03d}",
            placed=orders.placed_at[row],
            promise=promise[sim_id],
            measured={
                key: float(orders.measured[row, i]) for i, key in enumerate(INPUTS)
            },
            incident=orders.incidents[row],
            # The order's own streams, so they do not depend on the run's size.
            rng=np.random.default_rng([seed, position[sim_id], number]),
            planting=(
                np.random.default_rng([seed, position[sim_id], number, 1])
                if plant
                else None
            ),
        )
    timed.sort(key=lambda pair: pair[0])
    return [event for _, event in timed]
