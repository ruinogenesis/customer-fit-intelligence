"""Gaps and the asymmetric transform for the model inputs x1..x8."""

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


def gap(actual: float, expected: float, *, higher_is_better: bool = True) -> float:
    """What happened minus what was expected, signed so positive means better."""
    return actual - expected if higher_is_better else expected - actual


def input_gap(key: str, actual: float, expected: float) -> float:
    """gap() with the direction of input `key` (x1..x8)."""
    return gap(actual, expected, higher_is_better=HIGHER_IS_BETTER[key])


def asymmetric(value: float, kappa: float) -> float:
    """g(gap): a surplus passes unchanged; a shortfall is multiplied by kappa."""
    return value if value >= 0.0 else kappa * value


def transform(
    gaps: Mapping[str, float], kappas: Mapping[str, float]
) -> dict[str, float]:
    """Apply asymmetric() to each gap, with that input's own kappa."""
    return {key: asymmetric(value, kappas[key]) for key, value in gaps.items()}
