"""The v2 model: its fixed structure, its learned numbers and the numpy pass.

Theory sets the structure and data sets the weights. The masks below copy the
doc's layer tables: a 1 is a path the model may use, a 0 is a path it may
never use. The simulator, the PyTorch network and the numpy forward pass all
read them from here, so the structure is written down once.

ModelParams holds what training learned, as model.json stores it (T112).
forward() turns gaps into the five latents and outcomes() turns fit into the
three heads, with the same steps as fitcore.train's network, so the API can
score without PyTorch.

numpy only: the API imports this module and must never need PyTorch.
"""

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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


# The layout of model.json that ModelParams writes and reads.
MODEL_FORMAT = 1

Array = NDArray[np.float64]


def _frozen(name: str, value: ArrayLike, shape: tuple[int, ...]) -> Array:
    """A read-only float copy of value, checked for shape and finiteness."""
    array = np.array(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _number(name: str, value: object) -> float:
    """value as a float, checked to be a finite number.

    Raises TypeError unless value is a number (a bool is not), and
    ValueError unless it is finite.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return float(value)


@dataclass(frozen=True, eq=False)
class ModelParams:
    """What training learned, as model.json stores it: all the numpy pass needs.

    gap_scale and kappa: one value per input, x1..x8. masks, weights and
    biases: one per layer, in LAYERS order. A weight is its path's effective
    weight, softplus(raw) times the mask, never the raw value. Heads: the
    reorder logit is reorder_intercept + reorder_slope * fit, the churn logit
    is churn_intercept - churn_slope * fit, and P(NPS <= k) =
    sigmoid(nps_cuts[k] - nps_slope * fit) for k = 0..9.

    Arrays are copied and made read-only. Raises ValueError for a wrong
    shape, a value that is not finite, a mask that adds a path, a weight or
    head slope below zero, a weight off its mask, a kappa below 1, a scale of
    zero or less, or NPS cuts that do not rise; TypeError for a head number
    that is not a number.
    """

    gap_scale: Array
    kappa: Array
    masks: tuple[Array, ...]
    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    reorder_intercept: float
    reorder_slope: float
    churn_intercept: float
    churn_slope: float
    nps_slope: float
    nps_cuts: Array

    def __post_init__(self) -> None:
        count = len(INPUTS)
        gap_scale = _frozen("gap_scale", self.gap_scale, (count,))
        kappa = _frozen("kappa", self.kappa, (count,))
        if np.any(gap_scale <= 0.0):
            raise ValueError("gap_scale must be above zero for every input")
        if np.any(kappa < 1.0):
            raise ValueError("kappa must be at least 1 for every input")
        masks = check_masks(self.masks)
        if len(self.weights) != len(masks) or len(self.biases) != len(masks):
            raise ValueError(f"expected {len(masks)} weights and {len(masks)} biases")
        weights, biases = [], []
        layers = zip(masks, self.weights, self.biases, strict=True)
        for number, (mask, given_weight, given_bias) in enumerate(layers, start=1):
            weight = _frozen(f"layer {number} weight", given_weight, mask.shape)
            if np.any(weight < 0.0):
                raise ValueError(f"layer {number} weights must be zero or more")
            if np.any((weight != 0.0) & (mask == 0.0)):
                raise ValueError(f"layer {number} has a weight off its mask")
            weights.append(weight)
            biases.append(_frozen(f"layer {number} bias", given_bias, mask.shape[:1]))
        heads = {
            name: _number(name, getattr(self, name))
            for name in (
                "reorder_intercept",
                "reorder_slope",
                "churn_intercept",
                "churn_slope",
                "nps_slope",
            )
        }
        if min(heads["reorder_slope"], heads["churn_slope"], heads["nps_slope"]) < 0:
            raise ValueError("head slopes must be zero or more")
        cuts = _frozen("nps_cuts", self.nps_cuts, (10,))
        if np.any(np.diff(cuts) <= 0.0):
            raise ValueError("nps_cuts must rise strictly")
        checked = {
            "gap_scale": gap_scale,
            "kappa": kappa,
            "masks": masks,
            "weights": tuple(weights),
            "biases": tuple(biases),
            "nps_cuts": cuts,
            **heads,
        }
        for name, value in checked.items():
            object.__setattr__(self, name, value)

    def to_json(self) -> str:
        """model.json: this model as indented JSON text, ending in a newline.

        Python writes each float so that it reads back exactly, so
        from_json(to_json()) gives the same numbers, bit for bit.
        """
        layers = [
            {
                "outputs": list(outputs),
                "sources": list(sources),
                "mask": mask.astype(np.int64).tolist(),
                "weight": weight.tolist(),
                "bias": bias.tolist(),
            }
            for (outputs, sources), mask, weight, bias in zip(
                LAYERS, self.masks, self.weights, self.biases, strict=True
            )
        ]
        document = {
            "format": MODEL_FORMAT,
            "inputs": list(INPUTS),
            "latents": list(LATENTS),
            "gap_scale": self.gap_scale.tolist(),
            "kappa": self.kappa.tolist(),
            "layers": layers,
            "heads": {
                "reorder": {
                    "intercept": self.reorder_intercept,
                    "slope": self.reorder_slope,
                },
                "churn": {"intercept": self.churn_intercept, "slope": self.churn_slope},
                "nps": {"slope": self.nps_slope, "cuts": self.nps_cuts.tolist()},
            },
        }
        return json.dumps(document, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str | bytes) -> "ModelParams":
        """Read model.json, as to_json writes it, and check it.

        Raises ValueError unless text is JSON in format MODEL_FORMAT, its
        input, latent and layer names match this module's, and its numbers
        pass every check of ModelParams.
        """
        try:
            document = json.loads(text)
            if document["format"] != MODEL_FORMAT:
                raise ValueError(
                    f"model.json format must be {MODEL_FORMAT}, "
                    f"got {document['format']!r}"
                )
            layers, heads = document["layers"], document["heads"]
            names = [document["inputs"], document["latents"]] + [
                [layer["outputs"], layer["sources"]] for layer in layers
            ]
            expected = [list(INPUTS), list(LATENTS)] + [
                [list(outputs), list(sources)] for outputs, sources in LAYERS
            ]
            if names != expected:
                raise ValueError(
                    "model.json names must match fitcore's inputs, latents and layers"
                )
            return cls(
                gap_scale=document["gap_scale"],
                kappa=document["kappa"],
                masks=tuple(layer["mask"] for layer in layers),
                weights=tuple(layer["weight"] for layer in layers),
                biases=tuple(layer["bias"] for layer in layers),
                reorder_intercept=heads["reorder"]["intercept"],
                reorder_slope=heads["reorder"]["slope"],
                churn_intercept=heads["churn"]["intercept"],
                churn_slope=heads["churn"]["slope"],
                nps_slope=heads["nps"]["slope"],
                nps_cuts=heads["nps"]["cuts"],
            )
        except (KeyError, TypeError) as err:
            raise ValueError(
                f"model.json has a missing or wrong field: {err!r}"
            ) from err


def _sigmoid(x: Array) -> Array:
    """1 / (1 + e^-x), as torch.sigmoid computes it.

    For x below about -709, e^-x overflows to inf and the result is the
    right 0, so that overflow is not reported.
    """
    with np.errstate(over="ignore"):
        result: Array = 1.0 / (1.0 + np.exp(-x))
    return result


def forward(params: ModelParams, gaps: ArrayLike) -> Array:
    """The five latents (n x 5, in LATENTS order) for gaps (n x 8, own units).

    The same steps as fitcore.train.FitNetwork: the asymmetric transform with
    params.kappa, division by params.gap_scale, then each layer's
    sigmoid(sources @ weight.T + bias), in LAYERS order. Raises ValueError
    unless gaps has shape (n, 8) and every value is finite.
    """
    g = np.asarray(gaps, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != len(INPUTS):
        raise ValueError(f"gaps must have shape (n, {len(INPUTS)}), got {g.shape}")
    if not np.all(np.isfinite(g)):
        raise ValueError("gaps must be finite")
    z = np.where(g >= 0.0, g, g * params.kappa) / params.gap_scale
    values: dict[str, Array] = dict(zip(INPUTS, z.T, strict=True))
    for (outputs, sources), weight, bias in zip(
        LAYERS, params.weights, params.biases, strict=True
    ):
        x = np.column_stack([values[name] for name in sources])
        values.update(zip(outputs, _sigmoid(x @ weight.T + bias).T, strict=True))
    return np.column_stack([values[name] for name in LATENTS])


@dataclass(frozen=True, eq=False)
class Outcomes:
    """What the three heads say for each fit score.

    reorder: the chance of another order within 14 days. churn: the chance
    of no order in the next 30 days, the customer's churn_risk. nps: n x 11,
    the chance of each NPS answer, 0 to 10; each row sums to 1.
    """

    reorder: Array
    churn: Array
    nps: Array


def outcomes(params: ModelParams, fit: ArrayLike) -> Outcomes:
    """The three heads for each fit score, as fitcore.train.OutcomeHeads.

    fit: one score per row, e.g. forward(params, gaps)[:, -1]. Raises
    ValueError unless fit is one-dimensional and finite.
    """
    f = np.asarray(fit, dtype=np.float64)
    if f.ndim != 1 or not np.all(np.isfinite(f)):
        raise ValueError("fit must be one-dimensional and finite")
    at_most = _sigmoid(params.nps_cuts[None, :] - params.nps_slope * f[:, None])
    edges = np.column_stack([np.zeros(len(f)), at_most, np.ones(len(f))])
    return Outcomes(
        reorder=_sigmoid(params.reorder_intercept + params.reorder_slope * f),
        churn=_sigmoid(params.churn_intercept - params.churn_slope * f),
        nps=np.diff(edges, axis=1),
    )
