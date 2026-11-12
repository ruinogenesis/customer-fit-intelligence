"""Simulated orders: what each customer's orders deliver, in raw units.

Every customer has a service profile: their usual value of each measure,
how often an order goes wrong (an incident), and how often they order. One
order's measurements are the usual values plus a little order-to-order
noise, plus the effect of an incident when there is one.

Per-customer baselines (fitcore.expectations) soon learn a customer's usual
values, so steady differences between customers fade from their gaps.
Incidents stay surprising, so a customer's incident rate is the trait that
keeps them different from others once their expectations have adapted.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class Measure:
    """How one input's raw measurement varies, in its own units.

    usual: the population average, which is also the prior a new
    customer's expectation starts from. spread: how far customers' usual
    values differ (standard deviation). noise: how far one order strays
    from the customer's usual (standard deviation). low, high: the
    measurement's possible range.
    """

    usual: float
    spread: float
    noise: float
    low: float
    high: float


MEASURES: Mapping[str, Measure] = MappingProxyType(
    {
        # x1: delivery time, minutes
        "x1": Measure(usual=32.0, spread=6.0, noise=4.0, low=1.0, high=240.0),
        # x2: time to checkout, seconds
        "x2": Measure(usual=90.0, spread=20.0, noise=15.0, low=5.0, high=600.0),
        # x3: fee-to-basket ratio
        "x3": Measure(usual=0.12, spread=0.03, noise=0.01, low=0.0, high=1.0),
        # x4: share of items right
        "x4": Measure(usual=0.97, spread=0.01, noise=0.02, low=0.0, high=1.0),
        # x5: courier rating, 1 to 5
        "x5": Measure(usual=4.5, spread=0.25, noise=0.3, low=1.0, high=5.0),
        # x6: food rating, 1 to 5
        "x6": Measure(usual=4.3, spread=0.3, noise=0.3, low=1.0, high=5.0),
        # x7: recommendation hit rate
        "x7": Measure(usual=0.3, spread=0.1, noise=0.1, low=0.0, high=1.0),
        # x8: recovery score, 1 to 5
        "x8": Measure(usual=4.0, spread=0.4, noise=0.3, low=1.0, high=5.0),
    }
)

# Population priors for the inputs that use the customer's own history.
# x1 is judged against the promised delivery time instead.
PRIORS: Mapping[str, float] = MappingProxyType(
    {key: measure.usual for key, measure in MEASURES.items() if key != "x1"}
)

# How late a customer's deliveries usually are against the promise, minutes.
LATENESS_MEAN = 2.0
LATENESS_SPREAD = 4.0
PROMISE_LOW = 10.0

# An incident hits one input, by this much, in that input's own units.
INCIDENTS: Mapping[str, tuple[str, float]] = MappingProxyType(
    {
        "late": ("x1", 20.0),
        "wrong_items": ("x4", -0.3),
        "rude_courier": ("x5", -2.0),
        "cold_food": ("x6", -1.5),
        "poor_support": ("x8", -1.5),
    }
)

# Incident rate per customer: Beta(2, 18), so 10% of orders on average.
INCIDENT_RATE_A = 2.0
INCIDENT_RATE_B = 18.0

# Days between orders per customer: lognormal around a median of 7 days.
DAYS_MEDIAN = 7.0
DAYS_SIGMA = 0.5


@dataclass(frozen=True)
class ServiceProfile:
    """One customer's service: what they usually get, and how often it fails.

    promise: the promised delivery time in minutes, x1's expectation.
    usual: the customer's usual value of each measure, x1..x8, own units.
    incident_rate: the chance that any one order has an incident.
    days_between_orders: the average number of days from one order to the
    next.
    """

    customer_id: str
    promise: float
    usual: Mapping[str, float]
    incident_rate: float
    days_between_orders: float


def simulate_profiles(n: int, seed: int) -> tuple[ServiceProfile, ...]:
    """n service profiles. The same n and seed always give the same profiles.

    Raises ValueError if n is below 1.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(n):
        usual = {}
        for key, m in MEASURES.items():
            value = m.usual + m.spread * rng.standard_normal()
            usual[key] = float(np.clip(value, m.low, m.high))
        lateness = LATENESS_MEAN + LATENESS_SPREAD * rng.standard_normal()
        rate = rng.beta(INCIDENT_RATE_A, INCIDENT_RATE_B)
        days = rng.lognormal(np.log(DAYS_MEDIAN), DAYS_SIGMA)
        profiles.append(
            ServiceProfile(
                customer_id=f"sim-{i:06d}",
                promise=max(PROMISE_LOW, usual["x1"] - lateness),
                usual=MappingProxyType(usual),
                incident_rate=float(rate),
                days_between_orders=float(days),
            )
        )
    return tuple(profiles)


def draw_order(
    profile: ServiceProfile, rng: np.random.Generator
) -> tuple[dict[str, float], str | None]:
    """One order's raw measurements, x1..x8, and its incident (None if none).

    Each measure is the customer's usual value plus order-to-order noise,
    plus the incident's effect if this order has one, kept within the
    measure's possible range.
    """
    values = {
        key: profile.usual[key] + m.noise * rng.standard_normal()
        for key, m in MEASURES.items()
    }
    incident = None
    if rng.random() < profile.incident_rate:
        names = tuple(INCIDENTS)
        incident = names[int(rng.integers(len(names)))]
        key, effect = INCIDENTS[incident]
        values[key] += effect
    measured = {
        key: float(np.clip(value, MEASURES[key].low, MEASURES[key].high))
        for key, value in values.items()
    }
    return measured, incident
