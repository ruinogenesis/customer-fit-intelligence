"""Model cards: what a model version is, how well it does and what changed.

The doc asks for a card for every version: data window, metrics, masks and
the difference from the previous version (T113). ModelCard holds those and
writes them as markdown, to sit next to the version's model.json. The
retraining job (T701) and the Models screen (T407) reuse it.

Metrics are computed from model.json's numbers (fitcore.model), so a card
describes exactly the file the API serves. numpy only, like fitcore.model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fitcore.dataset import TrainingSet
from fitcore.features import INPUTS
from fitcore.model import LATENTS, PATHS, ModelParams, forward, outcomes
from fitcore.recovery import average_ranks

Array = NDArray[np.float64]


def _known(chance: ArrayLike, label: ArrayLike) -> tuple[Array, Array]:
    """The chances and 0/1 labels of the rows whose label is known (not NaN)."""
    p = np.asarray(chance, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    if p.ndim != 1 or p.shape != y.shape:
        raise ValueError("chance and label must be one-dimensional, of one length")
    known = ~np.isnan(y)
    if not known.any():
        raise ValueError("no row has a known label")
    p, y = p[known], y[known]
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("labels must be 0, 1 or NaN")
    return p, y


def log_loss(chance: ArrayLike, label: ArrayLike) -> float:
    """Mean cross-entropy of chances against 0/1 labels; NaN labels are skipped.

    Lower is better. Raises ValueError unless every chance lies strictly
    between 0 and 1 and at least one label is known.
    """
    p, y = _known(chance, label)
    if not np.all((p > 0.0) & (p < 1.0)):
        raise ValueError("every chance must lie strictly between 0 and 1")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def auc(score: ArrayLike, label: ArrayLike) -> float:
    """The chance that a random 1 scores above a random 0 (ties count half).

    0.5 is guessing and 1 is a perfect ranking. NaN labels are skipped.
    Raises ValueError unless both 0s and 1s are known.
    """
    s, y = _known(score, label)
    ones = y == 1.0
    n_ones, n_zeros = int(ones.sum()), int((~ones).sum())
    if n_ones == 0 or n_zeros == 0:
        raise ValueError("auc needs both 0s and 1s")
    # Ranks count from 0 (fitcore.recovery), so the 1s' rank sum, less its
    # least possible value n(n - 1) / 2, counts the (1, 0) pairs in order.
    ranks = average_ranks(s[:, None])[:, 0]
    in_order = ranks[ones].sum() - n_ones * (n_ones - 1) / 2
    return float(in_order / (n_ones * n_zeros))


@dataclass(frozen=True)
class DataWindow:
    """Where a model's data came from: the source, the dates and the split."""

    source: str
    first_order: date
    last_order: date
    train_customers: int
    train_orders: int
    test_customers: int
    test_orders: int


@dataclass(frozen=True)
class OutcomeResult:
    """How one head did on held-out orders.

    log_loss next to intercept_only, the loss of guessing without fit (lower
    is better); auc for reorder and churn, None for NPS; the mean predicted
    chance (or expected NPS) next to the observed rate (or mean answer).
    """

    outcome: str
    known: int
    log_loss: float
    intercept_only: float
    auc: float | None
    predicted: float
    observed: float


def outcome_results(
    params: ModelParams, test: TrainingSet, intercept_only: Mapping[str, float]
) -> tuple[OutcomeResult, ...]:
    """Reorder, churn and NPS on test, scored with the numpy forward pass.

    intercept_only: the test loss of guessing without fit, keyed reorder,
    churn and nps, as fitcore.train.intercept_only_losses gives it.
    """
    heads = outcomes(params, forward(params, test.gaps)[:, -1])
    results = []
    for name, key, chance, label in (
        ("reorder 14d", "reorder", heads.reorder, test.reorder),
        ("churn 30d", "churn", heads.churn, test.churn),
    ):
        p, y = _known(chance, label)
        results.append(
            OutcomeResult(
                outcome=name,
                known=len(y),
                log_loss=log_loss(chance, label),
                intercept_only=float(intercept_only[key]),
                auc=auc(chance, label),
                predicted=float(p.mean()),
                observed=float(y.mean()),
            )
        )
    known = ~np.isnan(test.nps)
    if not known.any():
        raise ValueError("test has no NPS answers")
    answers = test.nps[known].astype(np.intp)
    chances = heads.nps[known]
    results.append(
        OutcomeResult(
            outcome="NPS",
            known=len(answers),
            log_loss=float(-np.mean(np.log(chances[np.arange(len(answers)), answers]))),
            intercept_only=float(intercept_only["nps"]),
            auc=None,
            predicted=float(np.mean(chances @ np.arange(11))),
            observed=float(answers.mean()),
        )
    )
    return tuple(results)


@dataclass(frozen=True)
class Comparison:
    """The churn head against a simpler baseline, on the same held-out orders."""

    baseline: str
    known: int
    model_log_loss: float
    baseline_log_loss: float
    model_auc: float
    baseline_auc: float

    @property
    def model_wins(self) -> bool:
        """True when the model has the lower log loss and the higher AUC."""
        return (
            self.model_log_loss < self.baseline_log_loss
            and self.model_auc > self.baseline_auc
        )


def compare_churn(
    params: ModelParams, test: TrainingSet, baseline: str, chance: ArrayLike
) -> Comparison:
    """The churn head vs a baseline's churn chances for the same test rows.

    chance: one baseline chance per row of test, e.g. from order frequency
    alone (the doc's success target). Raises ValueError unless it has one
    value per row.
    """
    model = outcomes(params, forward(params, test.gaps)[:, -1]).churn
    theirs = np.asarray(chance, dtype=np.float64)
    if theirs.shape != model.shape:
        raise ValueError(f"chance must have one value per test row ({len(model)})")
    _, y = _known(model, test.churn)
    return Comparison(
        baseline=baseline,
        known=len(y),
        model_log_loss=log_loss(model, test.churn),
        baseline_log_loss=log_loss(theirs, test.churn),
        model_auc=auc(model, test.churn),
        baseline_auc=auc(theirs, test.churn),
    )


def weight_changes(
    params: ModelParams, previous: ModelParams
) -> tuple[tuple[str, float, float], ...]:
    """(name, before, now) for every path and the kappas, previous to params."""
    rows = [
        (
            path.name,
            float(previous.weights[path.layer - 1][path.position()]),
            float(params.weights[path.layer - 1][path.position()]),
        )
        for path in PATHS
    ]
    rows += [
        (f"kappa {name}", float(before), float(now))
        for name, before, now in zip(INPUTS, previous.kappa, params.kappa, strict=True)
    ]
    return tuple(rows)


# The columns of the metrics table, and the rule under its header.
_METRIC_COLUMNS = (
    "Outcome",
    "Known",
    "Log loss",
    "Intercept only",
    "AUC",
    "Predicted",
    "Observed",
)
_METRIC_RULE = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"


def _row(cells: Sequence[str]) -> str:
    """One markdown table row."""
    return "| " + " | ".join(cells) + " |"


@dataclass(frozen=True, eq=False)
class ModelCard:
    """One model version's card. to_markdown() writes it.

    results: from outcome_results. churn_baseline: from compare_churn.
    recovery: Pearson with the true latents, for simulated data only.
    previous: the model this one replaces, if any; its weights are compared.
    changes and limits: one plain sentence each.
    """

    title: str
    created: date
    params: ModelParams
    window: DataWindow
    results: Sequence[OutcomeResult]
    churn_baseline: Comparison | None = None
    recovery: Mapping[str, float] | None = None
    previous: ModelParams | None = None
    changes: Sequence[str] = ()
    limits: Sequence[str] = ()

    def to_markdown(self) -> str:
        """The card as markdown, ending in a newline."""
        w = self.window
        data = (
            f"Created {self.created.isoformat()}. Data: {w.source}, orders "
            f"{w.first_order.isoformat()} to {w.last_order.isoformat()}."
        )
        split = (
            f"Training: {w.train_customers:,} customers, {w.train_orders:,} orders. "
            f"Held out: {w.test_customers:,} customers, {w.test_orders:,} orders."
        )
        lines = [f"# Model card: {self.title}", "", data, split, ""]
        lines += ["## Held-out metrics", "", _row(_METRIC_COLUMNS), _METRIC_RULE]
        lines += [
            _row(
                (
                    r.outcome,
                    f"{r.known:,}",
                    f"{r.log_loss:.3f}",
                    f"{r.intercept_only:.3f}",
                    "" if r.auc is None else f"{r.auc:.3f}",
                    f"{r.predicted:.3f}",
                    f"{r.observed:.3f}",
                )
            )
            for r in self.results
        ]
        if self.churn_baseline is not None:
            c = self.churn_baseline
            verdict = "met" if c.model_wins else "not met"
            comparison = (
                f"Churn 30d against {c.baseline} ({c.known:,} orders): "
                f"log loss {c.model_log_loss:.3f} vs {c.baseline_log_loss:.3f}, "
                f"AUC {c.model_auc:.3f} vs {c.baseline_auc:.3f}. "
                f"Target (beat it on both): {verdict}."
            )
            lines += ["", comparison]
        if self.recovery is not None:
            found = [
                f"{name} {self.recovery[name]:.3f}"
                for name in LATENTS
                if name in self.recovery
            ]
            recovery = "Recovery (Pearson with the simulator's true latents): "
            lines += ["", recovery + ", ".join(found) + "."]
        lines += ["", "## Paths: masks and weights", ""]
        lines += [_row(("Path", "In use", "Weight")), "| --- | --- | ---: |"]
        for path in PATHS:
            layer = path.layer - 1
            row, column = path.position()
            in_use = "yes" if self.params.masks[layer][row, column] == 1.0 else "no"
            weight = self.params.weights[layer][row, column]
            lines.append(_row((path.name, in_use, f"{weight:.3f}")))
        kappa = self.params.kappa
        if np.all(kappa == kappa[0]):
            lines += ["", f"kappa, shared by all eight inputs: {kappa[0]:.3f}."]
        else:
            lines += ["", "kappa per input: " + ", ".join(f"{k:.3f}" for k in kappa)]
        lines += ["", "## Changes from the previous version", ""]
        if self.previous is None:
            lines.append("- First version: there is no previous model.")
        lines += [f"- {change}" for change in self.changes]
        if self.previous is not None:
            lines += ["", _row(("Weight", "Before", "Now")), "| --- | ---: | ---: |"]
            lines += [
                _row((name, f"{before:.3f}", f"{now:.3f}"))
                for name, before, now in weight_changes(self.params, self.previous)
                if before != now
            ]
        if self.limits:
            lines += ["", "## Limits and intended use", ""]
            lines += [f"- {limit}" for limit in self.limits]
        return "\n".join(lines) + "\n"
