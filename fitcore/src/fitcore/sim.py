"""Customer simulator: customers whose Love, Respect, Trust and Convenience are known.

The simulator hides a true model with exactly the structure the real model
will learn: the masks below copy the doc's layer tables, every weight is zero
or more, and every input has a kappa of at least 1. Gaps go in, true latents
come out. Training never sees this model; it has to recover the latents from
gaps, outcomes and anchoring indicators alone.

Gaps arrive in their own units. They pass through the asymmetric transform
(the same rule as fitcore.features), then are divided by a typical gap size
per input, so the true weights are comparable across inputs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fitcore.features import INPUTS

Array = NDArray[np.float64]

LAYER_1: tuple[str, ...] = ("respect", "love")
LAYER_2: tuple[str, ...] = ("convenience", "trust")
LAYER_2_INPUTS: tuple[str, ...] = ("respect", "love", "x1", "x2", "x3")


def _frozen(name: str, value: ArrayLike, shape: tuple[int, ...]) -> Array:
    """A read-only float copy of value, checked for shape and finiteness."""
    array = np.array(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


# Rows: respect, love. Columns: x1..x8. A 1 is a path the model may use.
MASK_1 = _frozen(
    "MASK_1",
    [
        [1, 0, 0, 1, 1, 1, 0, 0],  # respect <- x1, x4, x5, x6
        [0, 0, 0, 0, 0, 1, 1, 1],  # love <- x6, x7, x8
    ],
    (2, 8),
)

# Rows: convenience, trust. Columns: respect, love, x1, x2, x3.
MASK_2 = _frozen(
    "MASK_2",
    [
        [1, 0, 1, 1, 1],  # convenience <- respect, x1, x2, x3
        [1, 1, 0, 0, 0],  # trust <- respect, love
    ],
    (2, 5),
)


@dataclass(frozen=True, eq=False)
class TrueModel:
    """The hidden parameters that turn gaps into true latents.

    kappa and scale hold one value per input x1..x8. w1 and b1 make layer 1
    (respect, love); w2 and b2 make layer 2 (convenience, trust). Arrays are
    copied and made read-only. Raises ValueError for a wrong shape, a value
    that is not finite, kappa below 1, a scale of zero or less, a negative
    weight, or a weight outside its mask.
    """

    kappa: Array
    scale: Array
    w1: Array
    b1: Array
    w2: Array
    b2: Array

    def __post_init__(self) -> None:
        shapes = {
            "kappa": (8,),
            "scale": (8,),
            "w1": (2, 8),
            "b1": (2,),
            "w2": (2, 5),
            "b2": (2,),
        }
        for name, shape in shapes.items():
            object.__setattr__(self, name, _frozen(name, getattr(self, name), shape))
        if np.any(self.kappa < 1.0):
            raise ValueError("kappa must be at least 1 for every input")
        if np.any(self.scale <= 0.0):
            raise ValueError("scale must be positive for every input")
        for name, weights, mask in (("w1", self.w1, MASK_1), ("w2", self.w2, MASK_2)):
            if np.any(weights < 0.0):
                raise ValueError(f"{name} must be zero or more")
            if np.any((weights != 0.0) & (mask == 0.0)):
                raise ValueError(f"{name} has a weight outside its mask")


DEFAULT_TRUTH = TrueModel(
    # x1..x8
    kappa=np.array([2.0, 1.5, 2.0, 3.0, 1.5, 2.0, 1.2, 2.5]),
    # a typical gap per input, in its own units: minutes, seconds, fee ratio,
    # share of items right, rating points, rating points, hit rate, score
    scale=np.array([10.0, 30.0, 0.03, 0.1, 0.7, 0.7, 0.2, 1.0]),
    w1=np.array(
        [
            [0.8, 0.0, 0.0, 1.0, 0.6, 0.7, 0.0, 0.0],  # respect
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.7, 0.8],  # love
        ]
    ),
    b1=np.array([0.5, 0.0]),
    w2=np.array(
        [
            [2.0, 0.0, 0.5, 0.6, 0.8],  # convenience
            [2.5, 2.5, 0.0, 0.0, 0.0],  # trust
        ]
    ),
    b2=np.array([-1.0, -2.5]),
)


def apply_kappa(gaps: Array, kappa: Array) -> Array:
    """The asymmetric transform on a whole array: shortfalls times kappa."""
    return np.where(gaps >= 0.0, gaps, gaps * kappa)


def _sigmoid(x: Array) -> Array:
    """1 / (1 + e^-x), written so a large |x| neither overflows nor warns."""
    return np.exp(-np.logaddexp(0.0, -x))


def true_latents(gaps: ArrayLike, truth: TrueModel = DEFAULT_TRUTH) -> dict[str, Array]:
    """True latents for each row of gaps (n customers x 8 inputs, own units).

    Raises ValueError unless gaps has shape (n, 8) and every value is finite.
    """
    g = np.asarray(gaps, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != len(INPUTS):
        raise ValueError(f"gaps must have shape (n, {len(INPUTS)}), got {g.shape}")
    if not np.all(np.isfinite(g)):
        raise ValueError("gaps must be finite")
    z = apply_kappa(g, truth.kappa) / truth.scale
    layer1 = _sigmoid(z @ truth.w1.T + truth.b1)
    layer2 = _sigmoid(np.column_stack([layer1, z[:, :3]]) @ truth.w2.T + truth.b2)
    return {
        "respect": layer1[:, 0],
        "love": layer1[:, 1],
        "convenience": layer2[:, 0],
        "trust": layer2[:, 1],
    }
