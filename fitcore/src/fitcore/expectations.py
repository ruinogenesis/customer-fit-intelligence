"""Per-customer expectations: EWMA baselines with a cold-start prior.

Every model input x2..x8 is a gap between what happened and what *this*
customer expected. A Baseline holds one customer's habit for one input: an
exponentially weighted mean, the effective number of observations behind it,
and when it was last updated. Baselines live on the customer document and
are updated by the API; Colab runs the same code, so the two never disagree.

- update() folds in one observation. The first observations average evenly
  (alpha = 1, 1/2, 1/3, ...) until that would drop below ALPHA, after which
  recent observations keep weight ALPHA.
- expected() blends the habit with a population prior, s = w / (w + k): a
  new customer gets the prior, a regular gets their own habit.
- Both first decay the weight by a half-life, so a lapsed customer drifts
  back toward the prior without losing their habit. math.inf means never
  forget.

Timestamps must be timezone-aware. An observation older than last_seen still
counts but causes no forgetting, and last_seen never moves backwards.

x1 (speed vs promise) does not use this module: its expectation is the
promised ETA.
"""

import math
from dataclasses import dataclass
from datetime import datetime

ALPHA = 0.3
PRIOR_WEIGHT = 3.0
HALF_LIFE_DAYS = 60.0

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class Baseline:
    """One customer's habit for one input.

    mean: exponentially weighted mean of past observations.
    weight: effective number of observations behind it; decays with time.
    last_seen: when the last observation was folded in; None before the first.
    """

    mean: float = 0.0
    weight: float = 0.0
    last_seen: datetime | None = None


def _check_aware(at: datetime) -> None:
    if at.tzinfo is None or at.tzinfo.utcoffset(at) is None:
        raise ValueError("at must be timezone-aware")


def _check_half_life(half_life_days: float) -> None:
    if not half_life_days > 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")


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
    """Fold one observation into the baseline and return the new baseline.

    Raises ValueError for a naive `at`, alpha outside 0..1, or a
    non-positive half-life.
    """
    _check_aware(at)
    _check_half_life(half_life_days)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    weight = _decayed_weight(baseline, at, half_life_days)
    a = max(alpha, 1.0 / (weight + 1.0))
    mean = a * value + (1.0 - a) * baseline.mean
    last_seen = at if baseline.last_seen is None else max(at, baseline.last_seen)
    return Baseline(mean=mean, weight=weight + 1.0, last_seen=last_seen)


def expected(
    baseline: Baseline,
    prior: float,
    at: datetime,
    k: float = PRIOR_WEIGHT,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """The customer's expectation: their habit, shrunk toward the prior.

    Raises ValueError for a naive `at`, a k that is negative or NaN, or a
    non-positive half-life.
    """
    _check_aware(at)
    _check_half_life(half_life_days)
    if not k >= 0.0:
        raise ValueError(f"k must be zero or more, got {k}")
    weight = _decayed_weight(baseline, at, half_life_days)
    if weight == 0.0:
        return prior
    s = weight / (weight + k)
    return s * baseline.mean + (1.0 - s) * prior
