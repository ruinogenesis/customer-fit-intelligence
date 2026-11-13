"""Simulated behaviour: what each customer does that reflects their latents.

The doc anchors each latent with a survey item (fitcore.sim_survey) and one
or two behaviours. A behaviour describes a customer, not one order, so each
is drawn once per customer from their average true latent across their
orders, with noise, so it only partly reflects the latent:

- Love: favorites saved; the share of their orders placed without a promo.
- Respect: their average order rating, 1 to 5.
- Trust: the share of orders from a restaurant new to them; whether they
  saved a payment method.
- Convenience: their usual time from opening the app to checkout, in
  seconds; shorter means more convenient.

Each slope says how strongly a behaviour follows its latent; every slope is
zero or more, so a behaviour can only move the way the design says.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fitcore.sim import Array
from fitcore.sim_orders import Orders

# Each behaviour and the latent it anchors.
BEHAVIOURS: Mapping[str, str] = MappingProxyType(
    {
        "favorites_saved": "love",
        "promo_free_share": "love",
        "average_rating": "respect",
        "new_restaurant_share": "trust",
        "saved_payment": "trust",
        "checkout_seconds": "convenience",
    }
)

# A typical average latent: at these, each behaviour takes its typical value
# (3 favorites, half the orders without a promo, a 4.3 rating, a quarter of
# orders from new restaurants, 60% with a saved payment, 90 s to checkout).
CENTRES: Mapping[str, float] = MappingProxyType(
    {"love": 0.6, "respect": 0.65, "trust": 0.6, "convenience": 0.5}
)


@dataclass(frozen=True)
class TrueBehaviour:
    """How strongly each behaviour follows its latent (all zero or more).

    favorites: log of the expected count rises by this per unit of Love.
    promo_free, new_restaurants, saved_payment: the log-odds rise by this
    per unit of Love, Trust and Trust. rating: the expected rating rises by
    this per unit of Respect. checkout: the log of the time falls by this
    per unit of Convenience. Raises ValueError for a negative or
    non-finite slope.
    """

    favorites: float
    promo_free: float
    rating: float
    new_restaurants: float
    saved_payment: float
    checkout: float

    def __post_init__(self) -> None:
        for field in fields(self):
            slope = getattr(self, field.name)
            if not 0.0 <= slope < math.inf:
                raise ValueError(
                    f"{field.name} must be finite and zero or more, got {slope}"
                )


DEFAULT_BEHAVIOUR = TrueBehaviour(
    # tuned so each behaviour correlates about 0.45 to 0.7 with its latent
    favorites=10.0,
    promo_free=12.0,
    rating=3.5,
    new_restaurants=10.0,
    saved_payment=20.0,
    checkout=2.5,
)


@dataclass(frozen=True, eq=False)
class Behaviour:
    """One row per customer: their behaviours and their average true latents.

    values holds each behaviour in BEHAVIOURS; true_means holds each
    latent's average across the customer's orders, the truth the behaviours
    reflect. Arrays are read-only and line up with customer_ids.
    """

    customer_ids: tuple[str, ...]
    orders: NDArray[np.int64]
    values: Mapping[str, Array]
    true_means: Mapping[str, Array]

    def columns(self) -> dict[str, NDArray[Any]]:
        """One column per field, ready for pandas.DataFrame(...)."""
        cols: dict[str, NDArray[Any]] = {
            "customer_id": np.array(self.customer_ids),
            "orders": self.orders,
        }
        cols.update(self.values)
        cols.update({f"true_{k}": v for k, v in self.true_means.items()})
        return cols


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _read_only(values: list[float]) -> Array:
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def simulate_behaviour(
    orders: Orders, seed: int, behaviour: TrueBehaviour = DEFAULT_BEHAVIOUR
) -> Behaviour:
    """Each customer's behaviours, drawn from their average true latents.

    Customers come in the order they first appear in orders. Each gets their
    own random stream from (seed, position, 1); the extra 1 keeps it apart
    from the order loop's stream for the same customer.
    """
    ids = list(dict.fromkeys(orders.customer_ids))
    where = np.array(orders.customer_ids)
    b = behaviour
    counts: list[int] = []
    means: dict[str, list[float]] = {k: [] for k in orders.latents}
    values: dict[str, list[float]] = {k: [] for k in BEHAVIOURS}
    for position, customer in enumerate(ids):
        rng = np.random.default_rng([seed, position, 1])
        rows = where == customer
        n = int(rows.sum())
        counts.append(n)
        m = {k: float(v[rows].mean()) for k, v in orders.latents.items()}
        for k, v in m.items():
            means[k].append(v)
        d = {k: v - CENTRES[k] for k, v in m.items()}
        favorites = rng.poisson(3.0 * math.exp(b.favorites * d["love"]))
        promo_free = rng.binomial(n, _sigmoid(b.promo_free * d["love"])) / n
        rating = 4.3 + b.rating * d["respect"] + 0.3 * rng.normal()
        new = rng.binomial(n, _sigmoid(-1.1 + b.new_restaurants * d["trust"])) / n
        saved = rng.random() < _sigmoid(0.4 + b.saved_payment * d["trust"])
        noise = 0.3 * rng.normal()
        checkout = 90.0 * math.exp(-b.checkout * d["convenience"] + noise)
        values["favorites_saved"].append(float(favorites))
        values["promo_free_share"].append(promo_free)
        values["average_rating"].append(min(5.0, max(1.0, rating)))
        values["new_restaurant_share"].append(new)
        values["saved_payment"].append(float(saved))
        values["checkout_seconds"].append(checkout)
    order_counts = np.array(counts, dtype=np.int64)
    order_counts.setflags(write=False)
    return Behaviour(
        customer_ids=tuple(ids),
        orders=order_counts,
        values=MappingProxyType({k: _read_only(v) for k, v in values.items()}),
        true_means=MappingProxyType({k: _read_only(v) for k, v in means.items()}),
    )
