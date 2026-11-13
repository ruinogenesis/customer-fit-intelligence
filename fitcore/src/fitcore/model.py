"""The v2 model's fixed structure: latent names, layers and masks.

Theory sets the structure and data sets the weights. The masks below copy the
doc's layer tables: a 1 is a path the model may use, a 0 is a path it may
never use. The simulator, the PyTorch network and the numpy forward pass all
read them from here, so the structure is written down once.

numpy only: the API imports this module and must never need PyTorch.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
