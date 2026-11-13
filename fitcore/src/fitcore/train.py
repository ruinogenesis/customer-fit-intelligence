"""The v2 network in PyTorch, and how it is trained.

For Colab and the training job only. The API scores with numpy and never
imports this module, so PyTorch stays out of the API's container.

The structure comes from fitcore.model, and training cannot change it:

- every path weight is softplus(raw) times its mask, so it is above zero on
  the mask and exactly zero off it, after every step of any optimiser;
- every kappa is 1 + softplus(theta), so it stays above 1, as
  fitcore.features requires;
- gaps are divided by a fixed scale per input (a buffer, never trained),
  after the asymmetric transform. Biases are free.

Training (T109) fits the network against what can be observed, all held in
FitModel:

- three outcome heads read fit: reorder within 14 days and churn within 30
  days (logistic), and NPS 0 to 10 (cumulative logit). Their slopes are
  softplus, so reorder and NPS can only rise with fit, and churn only fall;
- anchors tie each latent to its survey item and its behaviours. The item
  is the latent's marker: its thresholds are fixed at k/7, which pins the
  latent's scale and centre (an answer of 4 sits around 0.5). Behaviours
  follow a + b * the customer's mean latent, with b above zero;
- a small L2 penalty on every slope: path weights, head slopes, item slopes
  and behaviour slopes. Intercepts, cuts and kappas are not penalised.

The loss adds up each term's mean over the rows that have its value, plus
the L2 penalty. fit_model trains with full-batch LBFGS until the loss stops
changing.

Everything is float64, so the numpy forward pass can match it exactly.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor, nn

from fitcore.dataset import ANCHOR_BEHAVIOURS, ITEM_LATENTS, TrainingSet, gap_scale
from fitcore.features import INPUTS
from fitcore.model import PATHS, THEORY_MASKS, Path, check_masks

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
    fit. Each latent is a sigmoid, so it lies between 0 and 1. masks: one per
    layer, the theory's by default. They may switch a path off, never add
    one (fitcore.model.check_masks raises ValueError otherwise).
    """

    def __init__(
        self, scale: ArrayLike, masks: Sequence[ArrayLike] = THEORY_MASKS
    ) -> None:
        super().__init__()
        mask_1, mask_2, mask_3 = check_masks(masks)
        self.gaps = AsymmetricGaps(scale)
        self.layer1 = PositiveMaskedLinear(mask_1)
        self.layer2 = PositiveMaskedLinear(mask_2)
        self.layer3 = PositiveMaskedLinear(mask_3)

    @property
    def layers(self) -> tuple[PositiveMaskedLinear, ...]:
        """Layers 1, 2 and 3."""
        return (self.layer1, self.layer2, self.layer3)

    def masks(self) -> tuple[np.ndarray, ...]:
        """The masks in use now, as numpy copies: 1 where a path is in use."""
        return tuple(layer.mask.numpy().copy() for layer in self.layers)

    def forward(self, gaps: Tensor) -> Tensor:
        z = self.gaps(gaps)
        layer1 = torch.sigmoid(self.layer1(z))
        layer2 = torch.sigmoid(self.layer2(torch.cat([layer1, z[:, :3]], dim=1)))
        layer3 = torch.sigmoid(self.layer3(layer2))
        return torch.cat([layer1, layer2, layer3], dim=1)


# Training settings (T109): see fit_model.
L2 = 1e-4
ROUNDS = 200
ITERATIONS_PER_ROUND = 20
TOLERANCE = 1e-9

# The parts of the loss, in the order loss_terms returns them.
TERMS: tuple[str, ...] = ("reorder", "churn", "nps", "items", "behaviours", "l2")

# Cuts this far out stand in for minus and plus infinity in ordinal_nll:
# sigmoid is exactly 0 or 1 there, and no infinity reaches a gradient.
_FAR = 1e4


def ordinal_nll(eta: Tensor, cuts: Tensor, answer: Tensor) -> Tensor:
    """Each answer's negative log-likelihood under a cumulative logit.

    Answers run from 0 to K - 1, and cuts holds K - 1 increasing numbers:
    P(answer <= k) = sigmoid(cuts[k] - eta). The chance of answer k is the
    gap between two of those, computed in logs so that it stays accurate far
    into the tails.
    """
    far = cuts.new_tensor([_FAR])
    edges = torch.cat([-far, cuts, far])
    k = answer.long()
    upper = F.logsigmoid(edges[k + 1] - eta)
    lower = F.logsigmoid(edges[k] - eta)
    return -(upper + torch.log(-torch.expm1(lower - upper)))


def _softplus_of(value: float) -> float:
    """The raw value whose softplus is value, as a plain float."""
    return float(inverse_softplus(torch.tensor(value, dtype=DTYPE)).item())


class OutcomeHeads(nn.Module):
    """The three heads that read fit.

    reorder: logit = intercept[0] + slope[0] * fit. churn: logit =
    intercept[1] - slope[1] * fit. NPS, 0 to 10: P(nps <= k) =
    sigmoid(cuts[k] - slope[2] * fit). Each slope is softplus(raw), so above
    zero: reorder and NPS rise with fit and churn falls, as theory says. The
    ten cuts are the first cut plus a running sum of softplus steps, so they
    always increase.
    """

    def __init__(self) -> None:
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(2, dtype=DTYPE))
        self.raw_slope = nn.Parameter(torch.full((3,), _softplus_of(1.0), dtype=DTYPE))
        self.first_cut = nn.Parameter(torch.tensor([-1.0], dtype=DTYPE))
        self.raw_steps = nn.Parameter(torch.full((9,), _softplus_of(0.5), dtype=DTYPE))

    @property
    def slope(self) -> Tensor:
        """The reorder, churn and NPS slopes, all above zero."""
        return F.softplus(self.raw_slope)

    @property
    def cuts(self) -> Tensor:
        """The ten NPS cuts, strictly increasing."""
        steps = torch.cumsum(F.softplus(self.raw_steps), dim=0)
        return torch.cat([self.first_cut, self.first_cut + steps])

    def forward(self, fit: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The reorder logit, the churn logit and the NPS eta for each fit."""
        slope = self.slope
        return (
            self.intercept[0] + slope[0] * fit,
            self.intercept[1] - slope[1] * fit,
            slope[2] * fit,
        )


class Anchors(nn.Module):
    """What ties Respect, Love, Convenience and Trust to observations.

    Survey items, one per latent in ITEM_LATENTS order: P(answer <= k) =
    sigmoid(s * (k/7 - latent)) for answers 1 to 7. The thresholds k/7 are
    fixed, so the item sets the latent's scale and centre; only s =
    softplus(raw), how closely answers follow the latent, is learned.

    Behaviours, one per ANCHOR_BEHAVIOURS, standardised with the training
    set's mean and sd (fixed buffers): a + b * the customer's mean latent,
    with b = softplus(raw) above zero. Raises ValueError unless mean and sd
    hold one finite value per behaviour and every sd is above zero.
    """

    behaviour_mean: Tensor
    behaviour_sd: Tensor
    behaviour_latent: Tensor
    thresholds: Tensor

    def __init__(self, behaviour_mean: ArrayLike, behaviour_sd: ArrayLike) -> None:
        super().__init__()
        count = len(ANCHOR_BEHAVIOURS)
        mean = torch.tensor(np.asarray(behaviour_mean, dtype=np.float64), dtype=DTYPE)
        sd = torch.tensor(np.asarray(behaviour_sd, dtype=np.float64), dtype=DTYPE)
        if mean.shape != (count,) or sd.shape != (count,):
            raise ValueError(f"behaviour mean and sd must have shape ({count},)")
        if not bool(torch.all(torch.isfinite(mean) & torch.isfinite(sd) & (sd > 0))):
            raise ValueError("behaviour mean and sd must be finite, sd above zero")
        latent = [ITEM_LATENTS.index(name) for name in ANCHOR_BEHAVIOURS.values()]
        self.register_buffer("behaviour_mean", mean)
        self.register_buffer("behaviour_sd", sd)
        self.register_buffer("behaviour_latent", torch.tensor(latent))
        self.register_buffer("thresholds", torch.arange(1, 7, dtype=DTYPE) / 7.0)
        start = _softplus_of(1.0)
        self.raw_item_slope = nn.Parameter(
            torch.full((len(ITEM_LATENTS),), start, dtype=DTYPE)
        )
        self.behaviour_intercept = nn.Parameter(torch.zeros(count, dtype=DTYPE))
        self.raw_behaviour_slope = nn.Parameter(
            torch.full((count,), start, dtype=DTYPE)
        )

    @property
    def item_slope(self) -> Tensor:
        """How closely each item's answers follow its latent, above zero."""
        return F.softplus(self.raw_item_slope)

    @property
    def behaviour_slope(self) -> Tensor:
        """How strongly each behaviour follows its latent, above zero."""
        return F.softplus(self.raw_behaviour_slope)


class FitModel(nn.Module):
    """Everything T109 trains: the network, the outcome heads and the anchors.

    scale: the gap scale (fitcore.dataset.gap_scale). behaviour_mean and
    behaviour_sd standardise the behaviours. All three come from the
    training set and stay fixed; for_training_set fills them in. masks: as
    for FitNetwork.
    """

    def __init__(
        self,
        scale: ArrayLike,
        behaviour_mean: ArrayLike,
        behaviour_sd: ArrayLike,
        masks: Sequence[ArrayLike] = THEORY_MASKS,
    ) -> None:
        super().__init__()
        self.network = FitNetwork(scale, masks)
        self.heads = OutcomeHeads()
        self.anchors = Anchors(behaviour_mean, behaviour_sd)

    @classmethod
    def for_training_set(
        cls, data: TrainingSet, masks: Sequence[ArrayLike] = THEORY_MASKS
    ) -> "FitModel":
        """A fresh model whose fixed numbers come from the training set."""
        behaviours = data.behaviours
        return cls(
            gap_scale(data.gaps),
            behaviours.mean(axis=0),
            behaviours.std(axis=0),
            masks,
        )

    def slopes(self) -> list[Tensor]:
        """Every slope the L2 penalty applies to."""
        network = self.network
        return [
            network.layer1.weight,
            network.layer2.weight,
            network.layer3.weight,
            self.heads.slope,
            self.anchors.item_slope,
            self.anchors.behaviour_slope,
        ]


@dataclass(frozen=True, eq=False)
class Batch:
    """A TrainingSet as float64 tensors, NaNs kept, made once for training."""

    gaps: Tensor
    customer: Tensor
    orders_per_customer: Tensor
    reorder: Tensor
    churn: Tensor
    nps: Tensor
    items: Tensor
    behaviours: Tensor

    @classmethod
    def from_training_set(cls, data: TrainingSet) -> "Batch":
        customer = torch.tensor(data.customer, dtype=torch.long)
        counts = torch.bincount(customer, minlength=len(data.customer_ids))
        return cls(
            gaps=torch.tensor(data.gaps, dtype=DTYPE),
            customer=customer,
            orders_per_customer=counts.to(DTYPE),
            reorder=torch.tensor(data.reorder, dtype=DTYPE),
            churn=torch.tensor(data.churn, dtype=DTYPE),
            nps=torch.tensor(data.nps, dtype=DTYPE),
            items=torch.tensor(data.items, dtype=DTYPE),
            behaviours=torch.tensor(data.behaviours, dtype=DTYPE),
        )


def _mean(values: Tensor) -> Tensor:
    """The mean, or 0 when there is nothing to average."""
    return values.sum() / max(values.numel(), 1)


def _logistic_loss(logit: Tensor, label: Tensor) -> Tensor:
    """Mean cross-entropy over the rows whose label is known."""
    known = ~torch.isnan(label)
    loss = F.binary_cross_entropy_with_logits(
        logit[known], label[known], reduction="none"
    )
    return _mean(loss)


def loss_terms(model: FitModel, batch: Batch, l2: float = L2) -> dict[str, Tensor]:
    """Each part of the loss, named as in TERMS. Training minimises their sum.

    Each data term is a mean over the rows where its value is known, so a
    NaN never counts, and a term with no known values adds 0. items adds up
    one mean per item, behaviours one mean per behaviour.
    """
    latents = model.network(batch.gaps)
    reorder, churn, nps = model.heads(latents[:, -1])
    anchors = model.anchors

    items = []
    for column in range(len(ITEM_LATENTS)):
        answer = batch.items[:, column]
        known = ~torch.isnan(answer)
        s = anchors.item_slope[column]
        eta = s * latents[known, column]
        items.append(_mean(ordinal_nll(eta, s * anchors.thresholds, answer[known] - 1)))

    customers = len(batch.orders_per_customer)
    total = torch.zeros(customers, len(ITEM_LATENTS), dtype=DTYPE)
    total = total.index_add(0, batch.customer, latents[:, : len(ITEM_LATENTS)])
    means = total / batch.orders_per_customer[:, None]
    predicted = (
        anchors.behaviour_intercept
        + anchors.behaviour_slope * means[:, anchors.behaviour_latent]
    )
    observed = (batch.behaviours - anchors.behaviour_mean) / anchors.behaviour_sd

    known = ~torch.isnan(batch.nps)
    return {
        "reorder": _logistic_loss(reorder, batch.reorder),
        "churn": _logistic_loss(churn, batch.churn),
        "nps": _mean(ordinal_nll(nps[known], model.heads.cuts, batch.nps[known])),
        "items": torch.stack(items).sum(),
        "behaviours": (predicted - observed).pow(2).mean(dim=0).sum(),
        "l2": l2 * sum(slope.pow(2).sum() for slope in model.slopes()),
    }


def evaluate(model: FitModel, data: TrainingSet) -> dict[str, float]:
    """Each data term of the loss on data, e.g. held-out customers (no L2)."""
    with torch.no_grad():
        terms = loss_terms(model, Batch.from_training_set(data), l2=0.0)
    return {name: float(value) for name, value in terms.items() if name != "l2"}


def intercept_only_losses(train: TrainingSet, test: TrainingSet) -> dict[str, float]:
    """What each head must beat: the test loss of guessing without fit.

    Reorder and churn get the training set's rate for every order; NPS gets
    the training set's share of each answer, 0 to 10.
    """
    losses: dict[str, float] = {}
    for name, rate_from, labels in (
        ("reorder", train.reorder, test.reorder),
        ("churn", train.churn, test.churn),
    ):
        rate = float(np.nanmean(rate_from))
        y = labels[~np.isnan(labels)]
        losses[name] = float(-np.mean(y * np.log(rate) + (1 - y) * np.log1p(-rate)))
    known = train.nps[~np.isnan(train.nps)].astype(np.intp)
    share = np.bincount(known, minlength=11) / len(known)
    answers = test.nps[~np.isnan(test.nps)].astype(np.intp)
    losses["nps"] = float(-np.mean(np.log(share[answers])))
    return losses


@dataclass(frozen=True)
class TrainingResult:
    """How training went: the loss before training and after each round."""

    losses: tuple[float, ...]
    converged: bool


def fit_model(
    model: FitModel,
    data: TrainingSet,
    *,
    l2: float = L2,
    rounds: int = ROUNDS,
    tolerance: float = TOLERANCE,
) -> TrainingResult:
    """Train model on data, in place, with full-batch LBFGS.

    Each round runs up to ITERATIONS_PER_ROUND LBFGS iterations with a
    strong-Wolfe line search. Training has converged when the loss changes
    by at most tolerance from one round to the next (relative to the loss,
    or absolute when the loss is below 1); it stops then, or after rounds
    rounds. Nothing is random, so the same model and data always give the
    same result.
    """
    batch = Batch.from_training_set(data)
    optimiser = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=ITERATIONS_PER_ROUND,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def loss() -> Tensor:
        return sum(loss_terms(model, batch, l2).values())

    def closure() -> Tensor:
        optimiser.zero_grad()
        value = loss()
        value.backward()
        return value

    with torch.no_grad():
        losses = [float(loss())]
    for _ in range(rounds):
        optimiser.step(closure)
        with torch.no_grad():
            losses.append(float(loss()))
        change = abs(losses[-2] - losses[-1])
        if change <= tolerance * max(1.0, abs(losses[-1])):
            return TrainingResult(tuple(losses), converged=True)
    return TrainingResult(tuple(losses), converged=False)


def without_path(model: FitModel, path: Path) -> FitModel:
    """A copy of model with one path switched off, ready to retrain.

    Everything else keeps its trained value, so retraining starts where the
    model ended (a warm start) and needs fewer rounds than starting afresh.
    From now on the path's weight is exactly 0 and gets no gradient. model
    itself is not changed. Raises ValueError unless model uses path.
    """
    if path not in PATHS:
        raise ValueError(f"the theory has no path {path.name}")
    ablated = deepcopy(model)
    layer = ablated.network.layers[path.layer - 1]
    row, column = path.position()
    if not bool(layer.mask[row, column] > 0.0):
        raise ValueError(f"the model does not use {path.name}")
    with torch.no_grad():
        layer.mask[row, column] = 0.0
    return ablated
