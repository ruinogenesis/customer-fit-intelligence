"""Gaps and the asymmetric transform for the model inputs x1..x8.

Every input is a gap between what happened and what this customer expected,
signed so that positive always means better than expected. That is what lets
every weight in the model be constrained to zero or more. HIGHER_IS_BETTER
records each input's direction: x1..x3 are times and costs, where less is
better; x4..x8 are qualities, where more is better.

A shortfall hurts more than an equal surplus helps, so each gap passes
through g(gap) = gap when gap >= 0 and kappa * gap when gap < 0, with
kappa >= 1; kappa = 1 means symmetric. model.json stores one kappa per
input. Training learns one value shared by all eight (T114: a kappa per
input did not earn its place), so today the eight are equal.

Gaps stay in their own units (minutes, ratios, ratings); any scaling is the
model's job. How x6..x8 combine their two measures into one number is
decided where they are measured, not here.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType

INPUTS: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")

HIGHER_IS_BETTER: Mapping[str, bool] = MappingProxyType(
    {
        "x1": False,  # delivery time vs the promised ETA
        "x2": False,  # time to checkout vs the customer's usual
        "x3": False,  # fee-to-basket ratio vs the customer's usual
        "x4": True,  # order fidelity
        "x5": True,  # courier rating
        "x6": True,  # food and selection
        "x7": True,  # personalization and delight
        "x8": True,  # recovery handling
    }
)


def _check_finite(name: str, x: float) -> None:
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")


def _check_input(key: str) -> None:
    if key not in HIGHER_IS_BETTER:
        raise ValueError(f"unknown input {key!r}, expected x1..x8")


def gap(actual: float, expected: float, *, higher_is_better: bool = True) -> float:
    """What happened minus what was expected, signed so positive means better.

    Raises ValueError if either value is NaN or infinite.
    """
    _check_finite("actual", actual)
    _check_finite("expected", expected)
    return actual - expected if higher_is_better else expected - actual


def input_gap(key: str, actual: float, expected: float) -> float:
    """gap() with the direction of input `key` (x1..x8).

    Raises ValueError for an unknown input or a value that is not finite.
    """
    _check_input(key)
    return gap(actual, expected, higher_is_better=HIGHER_IS_BETTER[key])


def asymmetric(value: float, kappa: float) -> float:
    """g(gap): a surplus passes unchanged; a shortfall is multiplied by kappa.

    Raises ValueError if the gap is not finite, or kappa is not a finite
    number of at least 1. kappa is checked even for a surplus, so a bad
    model fails on its first event rather than its first shortfall.
    """
    _check_finite("gap", value)
    if not 1.0 <= kappa < math.inf:
        raise ValueError(f"kappa must be finite and at least 1, got {kappa}")
    return value if value >= 0.0 else kappa * value


def transform(
    gaps: Mapping[str, float], kappas: Mapping[str, float]
) -> dict[str, float]:
    """Apply asymmetric() to each gap, with that input's own kappa.

    gaps may hold any subset of x1..x8, and kappas may cover more inputs
    than gaps. Raises ValueError for an unknown input or a missing kappa.
    """
    out: dict[str, float] = {}
    for key, value in gaps.items():
        _check_input(key)
        if key not in kappas:
            raise ValueError(f"no kappa for {key!r}")
        out[key] = asymmetric(value, kappas[key])
    return out
