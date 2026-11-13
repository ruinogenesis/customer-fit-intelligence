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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fitcore import expectations
from fitcore.features import INPUTS, input_gap
from fitcore.sim import DEFAULT_TRUTH, Array, TrueModel, true_latents
from fitcore.sim_outcomes import (
    DEFAULT_OUTCOMES,
    TrueOutcomes,
    churn_chance,
    fit,
    nps_answer,
    wait_factor,
)


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
        "wrong_items": ("x4", -0.15),
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


# The simulated period: half a year from the start of 2026.
START = datetime(2026, 1, 1, tzinfo=UTC)
DAYS = 180.0

# The outcome labels look this far ahead from each order.
REORDER_WINDOW = timedelta(days=14)
CHURN_WINDOW = timedelta(days=30)


@dataclass(frozen=True, eq=False)
class Orders:
    """Every simulated order, one row each: customer by customer, in time order.

    For each order: who placed it, its number in that customer's history
    (1 = first), when, and its incident (None if none). For each input
    x1..x8: the raw measurement, what the customer expected just before the
    order, and the gap between them. latents holds the true latents behind
    each order's gaps, and fit the true fit score. Arrays are read-only.

    Outcomes: nps is the order's survey answer, 0 to 10 (None if not
    surveyed); stopped is True on the order after which the customer quit.
    The labels are read from the history, as BigQuery will: reorder_14d
    (another order within 14 days) and churn_30d (no order in the next 30
    days). A label is None when its window runs past the simulated period.
    """

    customer_ids: tuple[str, ...]
    order_numbers: tuple[int, ...]
    placed_at: tuple[datetime, ...]
    incidents: tuple[str | None, ...]
    measured: Array
    expected: Array
    gaps: Array
    latents: Mapping[str, Array]
    fit: Array
    nps: tuple[int | None, ...]
    stopped: tuple[bool, ...]
    reorder_14d: tuple[bool | None, ...]
    churn_30d: tuple[bool | None, ...]

    def __len__(self) -> int:
        return len(self.customer_ids)

    def columns(self) -> dict[str, NDArray[Any]]:
        """One column per field, ready for pandas.DataFrame(...).

        Gaps are named x1..x8; raw values and expectations x1_measured,
        x1_expected and so on. An order without an incident shows "none";
        an empty answer or label is None.
        """
        cols: dict[str, NDArray[Any]] = {
            "customer_id": np.array(self.customer_ids),
            "order_number": np.array(self.order_numbers),
            "placed_at": np.array(self.placed_at),
            "incident": np.array([name or "none" for name in self.incidents]),
        }
        for i, key in enumerate(INPUTS):
            cols[f"{key}_measured"] = self.measured[:, i]
            cols[f"{key}_expected"] = self.expected[:, i]
            cols[key] = self.gaps[:, i]
        cols.update(self.latents)
        cols["fit"] = self.fit
        cols["nps"] = np.array(self.nps, dtype=object)
        cols["stopped"] = np.array(self.stopped)
        cols["reorder_14d"] = np.array(self.reorder_14d, dtype=object)
        cols["churn_30d"] = np.array(self.churn_30d, dtype=object)
        return cols


def _labels(
    times: list[datetime], end: datetime, window: timedelta, *, if_quiet: bool
) -> list[bool | None]:
    """One label per order of one customer, read from their order times.

    If the next order came within the window, the label is not if_quiet. If
    the whole window passed with no order, it is if_quiet. If the window
    runs past end before any next order, nobody can know yet: None.
    """
    labels: list[bool | None] = []
    for at, after in zip(times, [*times[1:], None], strict=True):
        if after is not None and after - at <= window:
            labels.append(not if_quiet)
        elif after is not None or at + window <= end:
            labels.append(if_quiet)
        else:
            labels.append(None)
    return labels


def _table(rows: list[list[float]]) -> Array:
    """Rows of eight values as a read-only (n, 8) array, even when n is 0."""
    table = np.array(rows, dtype=np.float64).reshape(-1, len(INPUTS))
    table.setflags(write=False)
    return table


def simulate_orders(
    profiles: Sequence[ServiceProfile],
    seed: int,
    start: datetime = START,
    days: float = DAYS,
    truth: TrueModel = DEFAULT_TRUTH,
    outcomes: TrueOutcomes = DEFAULT_OUTCOMES,
) -> Orders:
    """Every profile's order history from start, for the given number of days.

    For every order: draw its measurements; compare each with what the
    customer expected just before it (x1 with the promise, x2..x8 with their
    fitcore.expectations baseline, which starts from PRIORS); then let the
    order update those baselines. The order's true fit then decides what
    happens next (fitcore.sim_outcomes): maybe an NPS survey, maybe the
    customer stops for good, and otherwise how long until the next order,
    around days_between_orders.

    Each customer gets their own random stream, made from (seed, position),
    so a customer's history does not depend on how many others are
    simulated. Raises ValueError if days is not positive or start is naive.
    """
    if not days > 0:
        raise ValueError(f"days must be positive, got {days}")
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    end = start + timedelta(days=days)
    ids: list[str] = []
    numbers: list[int] = []
    times: list[datetime] = []
    incidents: list[str | None] = []
    measured: list[list[float]] = []
    wanted: list[list[float]] = []
    gaps: list[list[float]] = []
    scores: list[float] = []
    answers: list[int | None] = []
    stops: list[bool] = []
    reorders: list[bool | None] = []
    churns: list[bool | None] = []
    for position, profile in enumerate(profiles):
        rng = np.random.default_rng([seed, position])
        baselines = {key: expectations.Baseline() for key in PRIORS}
        at = start + timedelta(days=rng.exponential(profile.days_between_orders))
        mine: list[datetime] = []
        while at < end:
            values, incident = draw_order(profile, rng)
            expect = {"x1": profile.promise}
            for key, prior in PRIORS.items():
                expect[key] = expectations.expected(baselines[key], prior, at)
                baselines[key] = expectations.update(baselines[key], values[key], at)
            row = [input_gap(key, values[key], expect[key]) for key in INPUTS]
            latent = true_latents([row], truth)
            score = fit(latent["trust"][0], latent["convenience"][0], outcomes)
            surveyed = rng.random() < outcomes.survey_rate
            stopped = rng.random() < churn_chance(score, outcomes)
            ids.append(profile.customer_id)
            numbers.append(len(mine) + 1)
            times.append(at)
            mine.append(at)
            incidents.append(incident)
            measured.append([values[key] for key in INPUTS])
            wanted.append([expect[key] for key in INPUTS])
            gaps.append(row)
            scores.append(score)
            answers.append(nps_answer(score, rng, outcomes) if surveyed else None)
            stops.append(stopped)
            if stopped:
                break
            wait = profile.days_between_orders * wait_factor(score, outcomes)
            at += timedelta(days=rng.exponential(wait))
        reorders += _labels(mine, end, REORDER_WINDOW, if_quiet=False)
        churns += _labels(mine, end, CHURN_WINDOW, if_quiet=True)
    gap_table = _table(gaps)
    latents = true_latents(gap_table, truth)
    for column in latents.values():
        column.setflags(write=False)
    fit_column = np.array(scores, dtype=np.float64)
    fit_column.setflags(write=False)
    return Orders(
        customer_ids=tuple(ids),
        order_numbers=tuple(numbers),
        placed_at=tuple(times),
        incidents=tuple(incidents),
        measured=_table(measured),
        expected=_table(wanted),
        gaps=gap_table,
        latents=MappingProxyType(latents),
        fit=fit_column,
        nps=tuple(answers),
        stopped=tuple(stops),
        reorder_14d=tuple(reorders),
        churn_30d=tuple(churns),
    )
