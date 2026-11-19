"""Simulated outcomes: the hidden link from latents to fit to behaviour.

Fit = sigmoid(w_trust * trust + w_convenience * convenience + b), as in the
doc. Fit then changes what a customer does after an order:

- the wait until their next order: higher fit, shorter wait;
- the chance they stop ordering for good (churn): lower fit, higher chance;
- their answer to an NPS survey, 0 to 10: higher fit, higher answers.

Every weight and slope is zero or more, so these effects can only point the
way the design says. The reorder and churn labels are not drawn here: they
are read from the order history that this behaviour produces.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

# Fit at which the wait is the customer's usual and churn is at its base rate.
FIT_MIDPOINT = 0.5


@dataclass(frozen=True)
class TrueOutcomes:
    """The hidden settings that turn latents into fit, and fit into behaviour.

    w_fit, b_fit: Fit = sigmoid(w_fit[0] * trust + w_fit[1] * convenience
    + w_interaction * trust * convenience + b_fit); w_interaction is 0 unless
    a robustness check (T803) asks for an effect CFI's additive layer lacks. pace: the wait to the next order is the customer's usual
    days_between_orders times exp(-pace * (fit - FIT_MIDPOINT)).
    churn_base, churn_slope: after each order, the chance of stopping for
    good is sigmoid(churn_base - churn_slope * (fit - FIT_MIDPOINT)).
    survey_rate: the share of orders that get an NPS question. nps_slope,
    nps_cuts: an answer is how many of the ten cuts lie below
    nps_slope * fit plus logistic noise, so 0 to 10.

    Raises ValueError for a negative weight, pace or slope, a value that is
    not finite, a survey rate outside 0..1, or cuts that are not ten
    strictly increasing numbers.
    """

    w_fit: tuple[float, float]
    b_fit: float
    pace: float
    churn_base: float
    churn_slope: float
    survey_rate: float
    nps_slope: float
    nps_cuts: tuple[float, ...]
    w_interaction: float = 0.0

    def __post_init__(self) -> None:
        numbers = [
            *self.w_fit,
            self.b_fit,
            self.pace,
            self.churn_base,
            self.churn_slope,
            self.survey_rate,
            self.nps_slope,
            *self.nps_cuts,
            self.w_interaction,
        ]
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("every outcome setting must be finite")
        if len(self.w_fit) != 2 or min(self.w_fit) < 0.0:
            raise ValueError("w_fit must be two weights of zero or more")
        if min(self.pace, self.churn_slope, self.nps_slope) < 0.0:
            raise ValueError("pace, churn_slope and nps_slope must be zero or more")
        if not 0.0 <= self.survey_rate <= 1.0:
            raise ValueError(
                f"survey_rate must be between 0 and 1, got {self.survey_rate}"
            )
        cuts = self.nps_cuts
        if len(cuts) != 10 or any(a >= b for a, b in pairwise(cuts)):
            raise ValueError("nps_cuts must be ten strictly increasing numbers")


DEFAULT_OUTCOMES = TrueOutcomes(
    # with these, fit averages about 0.56 across simulated orders
    w_fit=(5.0, 3.0),
    b_fit=-4.3,
    # fit 0.8 waits about 0.55x as long as usual; fit 0.2 about 1.8x
    pace=2.0,
    # stopping after an order: 1.8% at fit 0.5, 10% at fit 0.2, 0.3% at 0.8
    churn_base=-4.0,
    churn_slope=6.0,
    survey_rate=0.2,
    # at fit 0.55: about 20% detractors (0-6), 35% passives, 45% promoters
    nps_slope=6.0,
    nps_cuts=(-0.6, -0.2, 0.1, 0.5, 0.9, 1.3, 1.9, 2.7, 3.5, 4.4),
)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fit(
    trust: float, convenience: float, outcomes: TrueOutcomes = DEFAULT_OUTCOMES
) -> float:
    """The true fit score, between 0 and 1."""
    w_trust, w_convenience = outcomes.w_fit
    pair = outcomes.w_interaction * trust * convenience
    return _sigmoid(
        w_trust * trust + w_convenience * convenience + pair + outcomes.b_fit
    )


def wait_factor(score: float, outcomes: TrueOutcomes = DEFAULT_OUTCOMES) -> float:
    """How much longer (above 1) or shorter (below 1) than usual the next wait is."""
    return math.exp(-outcomes.pace * (score - FIT_MIDPOINT))


def churn_chance(score: float, outcomes: TrueOutcomes = DEFAULT_OUTCOMES) -> float:
    """The chance that the customer stops ordering for good after this order."""
    return _sigmoid(outcomes.churn_base - outcomes.churn_slope * (score - FIT_MIDPOINT))


def nps_chances(
    score: float, outcomes: TrueOutcomes = DEFAULT_OUTCOMES
) -> tuple[float, ...]:
    """The chance of each NPS answer, 0 to 10, at this fit score. They sum to 1."""
    centre = outcomes.nps_slope * score
    at_most = [_sigmoid(cut - centre) for cut in outcomes.nps_cuts] + [1.0]
    return (at_most[0], *(b - a for a, b in pairwise(at_most)))


def nps_answer(
    score: float, rng: np.random.Generator, outcomes: TrueOutcomes = DEFAULT_OUTCOMES
) -> int:
    """One NPS answer, 0 to 10: how many cuts lie below the fit score plus noise."""
    u = outcomes.nps_slope * score + rng.logistic()
    return sum(1 for cut in outcomes.nps_cuts if cut < u)
