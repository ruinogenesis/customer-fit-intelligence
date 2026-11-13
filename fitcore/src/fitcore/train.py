"""The v2 network in PyTorch: masked, sign-constrained and trainable.

For Colab and the training job only. The API scores with numpy and never
imports this module, so PyTorch stays out of the API's container.

The structure comes from fitcore.model, and training cannot change it:

- every path weight is softplus(raw) times its mask, so it is above zero on
  the mask and exactly zero off it, after every step of any optimiser;
- every kappa is 1 + softplus(theta), so it stays above 1, as
  fitcore.features requires;
- gaps are divided by a fixed scale per input (a buffer, never trained),
  after the asymmetric transform. Biases are free.

Everything is float64, so the numpy forward pass can match it exactly.
"""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor, nn

from fitcore.features import INPUTS
from fitcore.model import MASK_1, MASK_2, MASK_3

DTYPE = torch.float64


def inverse_softplus(y: Tensor) -> Tensor:
    """The raw value whose softplus is y, for y > 0: y + log(1 - e^-y)."""
    return y + torch.log(-torch.expm1(-y))


class PositiveMaskedLinear(nn.Module):
    """A linear layer whose weights are softplus(raw) times a fixed 0/1 mask.

    Rows are outputs, columns are inputs, as in the mask. A weight on the
    mask is above zero and one off the mask is exactly zero, so no training
    step can add a path or flip a sign. The bias has no constraint.
    """

    mask: Tensor

    def __init__(self, mask: ArrayLike, init_weight: float = 0.5) -> None:
        super().__init__()
        m = torch.tensor(np.asarray(mask), dtype=DTYPE)
        self.register_buffer("mask", m)
        start = inverse_softplus(torch.tensor(init_weight, dtype=DTYPE))
        self.raw = nn.Parameter(torch.full(m.shape, start.item(), dtype=DTYPE))
        self.bias = nn.Parameter(torch.zeros(m.shape[0], dtype=DTYPE))

    @property
    def weight(self) -> Tensor:
        """The weights the layer uses: >= 0 everywhere, 0 off the mask."""
        return F.softplus(self.raw) * self.mask

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


class AsymmetricGaps(nn.Module):
    """g(gap) = gap if gap >= 0, else kappa * gap; then divided by scale.

    kappa = 1 + softplus(theta) is learned, one per input, always above 1.
    scale is fixed: a scale and the weights after it multiply each other, so
    they cannot both be learned. Raises ValueError unless scale holds one
    finite, positive value per input.
    """

    scale: Tensor

    def __init__(self, scale: ArrayLike, init_kappa: float = 1.5) -> None:
        super().__init__()
        s = torch.tensor(np.asarray(scale, dtype=np.float64), dtype=DTYPE)
        if s.shape != (len(INPUTS),):
            raise ValueError(f"scale must have shape ({len(INPUTS)},), got {s.shape}")
        if not bool(torch.all(torch.isfinite(s) & (s > 0.0))):
            raise ValueError("scale must be finite and positive for every input")
        if not init_kappa > 1.0:
            raise ValueError(f"init_kappa must be above 1, got {init_kappa}")
        self.register_buffer("scale", s)
        start = inverse_softplus(torch.tensor(init_kappa - 1.0, dtype=DTYPE))
        self.theta = nn.Parameter(torch.full(s.shape, start.item(), dtype=DTYPE))

    @property
    def kappa(self) -> Tensor:
        """One kappa per input, always above 1."""
        return 1.0 + F.softplus(self.theta)

    def forward(self, gaps: Tensor) -> Tensor:
        shaped = torch.where(gaps >= 0.0, gaps, gaps * self.kappa)
        return shaped / self.scale


class FitNetwork(nn.Module):
    """Gaps (n x 8, own units) in; the five latents (n x 5) out.

    Columns follow fitcore.model.LATENTS: respect, love, convenience, trust,
    fit. Each latent is a sigmoid, so it lies between 0 and 1. The outcome
    heads (reorder, churn, NPS) read fit and are added in training.
    """

    def __init__(self, scale: ArrayLike) -> None:
        super().__init__()
        self.gaps = AsymmetricGaps(scale)
        self.layer1 = PositiveMaskedLinear(MASK_1)
        self.layer2 = PositiveMaskedLinear(MASK_2)
        self.layer3 = PositiveMaskedLinear(MASK_3)

    def forward(self, gaps: Tensor) -> Tensor:
        z = self.gaps(gaps)
        layer1 = torch.sigmoid(self.layer1(z))
        layer2 = torch.sigmoid(self.layer2(torch.cat([layer1, z[:, :3]], dim=1)))
        layer3 = torch.sigmoid(self.layer3(layer2))
        return torch.cat([layer1, layer2, layer3], dim=1)
