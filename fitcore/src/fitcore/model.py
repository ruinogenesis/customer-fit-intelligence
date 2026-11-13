"""The v2 model's fixed structure: latent names, layers and masks.

Theory sets the structure and data sets the weights. The masks below copy the
doc's layer tables: a 1 is a path the model may use, a 0 is a path it may
never use. The simulator, the PyTorch network and the numpy forward pass all
read them from here, so the structure is written down once.

numpy only: the API imports this module and must never need PyTorch.
"""

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fitcore.features import INPUTS

LAYER_1: tuple[str, ...] = ("respect", "love")
LAYER_2: tuple[str, ...] = ("convenience", "trust")
LAYER_2_INPUTS: tuple[str, ...] = ("respect", "love", "x1", "x2", "x3")
LAYER_3: tuple[str, ...] = ("fit",)
LAYER_3_INPUTS: tuple[str, ...] = LAYER_2

# Every latent, in the order the network outputs them.
LATENTS: tuple[str, ...] = LAYER_1 + LAYER_2 + LAYER_3


def _mask(rows: ArrayLike, shape: tuple[int, int]) -> NDArray[np.float64]:
    """A read-only float mask, checked for shape and for holding only 0 and 1."""
    mask = np.array(rows, dtype=np.float64)
    if mask.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {mask.shape}")
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError("mask must hold only 0 and 1")
    mask.setflags(write=False)
    return mask


# Rows: respect, love. Columns: x1..x8.
MASK_1 = _mask(
    [
        [1, 0, 0, 1, 1, 1, 0, 0],  # respect <- x1, x4, x5, x6
        [0, 0, 0, 0, 0, 1, 1, 1],  # love <- x6, x7, x8
    ],
    (2, 8),
)

# Rows: convenience, trust. Columns: respect, love, x1, x2, x3.
MASK_2 = _mask(
    [
        [1, 0, 1, 1, 1],  # convenience <- respect, x1, x2, x3
        [1, 1, 0, 0, 0],  # trust <- respect, love
    ],
    (2, 5),
)

# Rows: fit. Columns: convenience, trust.
MASK_3 = _mask([[1, 1]], (1, 2))

# The theory's three masks, layer 1 first, and each layer's rows and columns.
THEORY_MASKS: tuple[NDArray[np.float64], ...] = (MASK_1, MASK_2, MASK_3)
LAYERS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (LAYER_1, INPUTS),
    (LAYER_2, LAYER_2_INPUTS),
    (LAYER_3, LAYER_3_INPUTS),
)


class Path(NamedTuple):
    """One connection the theory allows: output <- source, in layer 1, 2 or 3."""

    layer: int
    output: str
    source: str

    @property
    def name(self) -> str:
        return f"{self.output} <- {self.source}"

    def position(self) -> tuple[int, int]:
        """The path's row and column in its layer's mask."""
        outputs, sources = LAYERS[self.layer - 1]
        return outputs.index(self.output), sources.index(self.source)


# Every path the theory allows, layer by layer, in mask order: 7 + 6 + 2.
PATHS: tuple[Path, ...] = tuple(
    Path(layer, outputs[row], sources[column])
    for layer, ((outputs, sources), mask) in enumerate(
        zip(LAYERS, THEORY_MASKS, strict=True), start=1
    )
    for row, column in zip(*np.nonzero(mask), strict=True)
)


def check_masks(masks: Sequence[ArrayLike]) -> tuple[NDArray[np.float64], ...]:
    """Read-only copies of three masks that keep some of the theory's paths.

    A mask may switch off a path the theory allows, never add one. Raises
    ValueError unless there are three masks with the theory's shapes, holding
    only 0 and 1, and every 1 is a 1 in the theory's mask.
    """
    if len(masks) != len(THEORY_MASKS):
        raise ValueError(f"expected {len(THEORY_MASKS)} masks, got {len(masks)}")
    checked = []
    for given, theory in zip(masks, THEORY_MASKS, strict=True):
        mask = _mask(given, theory.shape)
        if np.any(mask > theory):
            raise ValueError("a mask may only remove paths, never add one")
        checked.append(mask)
    return tuple(checked)


def masks_without(paths: Iterable[Path]) -> tuple[NDArray[np.float64], ...]:
    """The theory's masks with the given paths switched off.

    Raises ValueError for a path the theory does not have.
    """
    masks = [np.array(mask) for mask in THEORY_MASKS]
    for path in paths:
        if path not in PATHS:
            raise ValueError(f"the theory has no path {path.name}")
        masks[path.layer - 1][path.position()] = 0.0
    return check_masks(masks)
