"""Simulated survey questions: NPS and one 1-to-7 item per latent.

The doc's survey plan: after some orders, the customer gets a single in-app
question, rotated so that no customer sees the same item twice in a row.
QUESTIONS is that rotation: NPS (answered from fit, in fitcore.sim_outcomes)
and four items that each anchor one latent.

An item's answer reflects the true latent at that order, with noise: how
many of six cuts lie below slope * latent plus logistic noise, plus 1, so
1 to 7. The slope sets how strongly answers follow the latent; the default
gives a correlation of about 0.6, as real survey items tend to have.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np

# The rotation of survey questions, in order.
QUESTIONS: tuple[str, ...] = ("nps", "love", "respect", "trust", "convenience")

# The wording of each latent's item, from the doc's anchoring table.
ITEM_TEXT: Mapping[str, str] = MappingProxyType(
    {
        "love": "I'd miss this app if it closed",
        "respect": "They get my order right",
        "trust": "I'd trust them with a big order",
        "convenience": "Ordering here is effortless",
    }
)


@dataclass(frozen=True)
class TrueItems:
    """How a latent turns into a 1-to-7 answer.

    An answer is 1 plus how many of the six cuts lie below slope * latent
    plus logistic noise. Raises ValueError for a negative or non-finite
    slope, or cuts that are not six strictly increasing finite numbers.
    """

    slope: float
    cuts: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.slope < math.inf:
            raise ValueError(f"slope must be finite and zero or more, got {self.slope}")
        cuts = self.cuts
        if (
            len(cuts) != 6
            or not all(math.isfinite(c) for c in cuts)
            or any(a >= b for a, b in pairwise(cuts))
        ):
            raise ValueError("cuts must be six strictly increasing finite numbers")


DEFAULT_ITEMS = TrueItems(
    # answers follow the latent with a correlation of about 0.6
    slope=12.0,
    # at a latent of 0.6, answers lean positive: mostly 5 to 7
    cuts=(3.7, 4.5, 5.2, 6.1, 7.0, 8.1),
)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def item_chances(latent: float, items: TrueItems = DEFAULT_ITEMS) -> tuple[float, ...]:
    """The chance of each answer, 1 to 7, at this latent. They sum to 1."""
    centre = items.slope * latent
    at_most = [_sigmoid(cut - centre) for cut in items.cuts] + [1.0]
    return (at_most[0], *(b - a for a, b in pairwise(at_most)))


def item_answer(
    latent: float, rng: np.random.Generator, items: TrueItems = DEFAULT_ITEMS
) -> int:
    """One answer, 1 to 7: 1 plus how many cuts lie below the latent plus noise."""
    u = items.slope * latent + rng.logistic()
    return 1 + sum(1 for cut in items.cuts if cut < u)
