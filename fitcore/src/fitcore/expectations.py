"""Per-customer expectations: EWMA baselines with a cold-start prior."""

import math
from dataclasses import dataclass
from datetime import datetime

ALPHA = 0.3
PRIOR_WEIGHT = 3.0
HALF_LIFE_DAYS = 60.0

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class Baseline:
    """One customer's habit for one input."""

    mean: float = 0.0
    weight: float = 0.0
    last_seen: datetime | None = None


def _decayed_weight(baseline: Baseline, at: datetime, half_life_days: float) -> float:
    """The weight after forgetting since last_seen; 0 for an empty baseline."""
    if baseline.last_seen is None:
        return 0.0
    days = max(0.0, (at - baseline.last_seen).total_seconds() / _SECONDS_PER_DAY)
    return baseline.weight * math.pow(0.5, days / half_life_days)


def update(
    baseline: Baseline,
    value: float,
    at: datetime,
    alpha: float = ALPHA,
    half_life_days: float = HALF_LIFE_DAYS,
) -> Baseline:
    """Fold one observation into the baseline and return the new baseline."""
    weight = _decayed_weight(baseline, at, half_life_days)
    a = max(alpha, 1.0 / (weight + 1.0))
    mean = a * value + (1.0 - a) * baseline.mean
    return Baseline(mean=mean, weight=weight + 1.0, last_seen=at)


def expected(
    baseline: Baseline,
    prior: float,
    at: datetime,
    k: float = PRIOR_WEIGHT,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """The customer's expectation: their habit, shrunk toward the prior."""
    weight = _decayed_weight(baseline, at, half_life_days)
    s = weight / (weight + k)
    return s * baseline.mean + (1.0 - s) * prior
