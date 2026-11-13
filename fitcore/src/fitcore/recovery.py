"""How well a trained model recovers known customer states.

On simulated data the true latents are known, so a trained model can be
checked against them. Two things are compared:

- the latents: the correlation between each predicted latent and its true
  value, per order and per customer average. A recovered latent's scale is
  set by its survey marker, not by the simulator, so recovery means a high
  correlation, not equal values. Pearson reads the values; Spearman reads
  only their order, so a stretched scale does not count against it;
- kappa: how much more a shortfall hurts than a surplus helps. It is a ratio
  within one input, so it does not depend on any scale and can be compared
  with the truth directly.

numpy only, so it runs and is tested without PyTorch.
"""

import numpy as np
from numpy.typing import ArrayLike

from fitcore.features import INPUTS
from fitcore.model import LATENTS
from fitcore.sim import Array

# What latent_recovery reports for each latent.
RECOVERY_MEASURES: tuple[str, ...] = (
    "pearson",
    "spearman",
    "customer_pearson",
    "customer_spearman",
)


def _columns(name: str, values: ArrayLike, width: int) -> Array:
    """values as a finite float array of shape (n, width), n of 2 or more."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape (n, {width}) with n >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _pair(a: ArrayLike, b: ArrayLike) -> tuple[Array, Array]:
    """a and b as finite float arrays of the same shape (n, k), n of 2 or more."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] < 2:
        raise ValueError("a and b must have the same shape (n, k) with n >= 2")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("a and b must be finite")
    return x, y


def pearson(a: ArrayLike, b: ArrayLike) -> Array:
    """The Pearson correlation of each column of a with the same column of b.

    Raises ValueError unless a and b are finite, have the same shape (n, k)
    with n of 2 or more, and no column is constant.
    """
    x, y = _pair(a, b)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    spread = np.sqrt((x**2).sum(axis=0) * (y**2).sum(axis=0))
    if np.any(spread == 0.0):
        raise ValueError("no column may be constant")
    result: Array = (x * y).sum(axis=0) / spread
    return result


def average_ranks(values: ArrayLike) -> Array:
    """The rank of each value in its column, 0 for the smallest.

    Tied values share the average of the ranks they span, so [10, 20, 20, 5]
    ranks as [1, 2.5, 2.5, 0].
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must have shape (n, k)")
    ranks = np.empty_like(array)
    for column in range(array.shape[1]):
        x = array[:, column]
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        ends = np.r_[starts[1:], len(x)]
        ranks[order, column] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return ranks


def spearman(a: ArrayLike, b: ArrayLike) -> Array:
    """The Spearman correlation of each column: Pearson on average ranks.

    It is 1 whenever one column only ever rises with the other, whatever the
    shape. Raises ValueError as pearson does.
    """
    x, y = _pair(a, b)
    return pearson(average_ranks(x), average_ranks(y))


def customer_means(values: ArrayLike, customer: ArrayLike) -> Array:
    """Each customer's average row: one row per customer, 0 to max(customer).

    Raises ValueError unless values is (n, k), customer holds n positions
    of 0 or more, and every customer up to the largest has a row.
    """
    array = np.asarray(values, dtype=np.float64)
    who = np.asarray(customer)
    if array.ndim != 2 or who.shape != (array.shape[0],):
        raise ValueError("values must be (n, k) and customer must have n entries")
    if len(who) == 0 or not np.issubdtype(who.dtype, np.integer) or who.min() < 0:
        raise ValueError("customer must hold positions 0, 1, 2, ...")
    counts = np.bincount(who)
    if np.any(counts == 0):
        raise ValueError("every customer up to the largest must have a row")
    totals = np.zeros((len(counts), array.shape[1]))
    np.add.at(totals, who, array)
    result: Array = totals / counts[:, None]
    return result


def latent_recovery(
    predicted: ArrayLike, truth: ArrayLike, customer: ArrayLike
) -> dict[str, Array]:
    """How closely each predicted latent follows the truth.

    predicted and truth: one row per order, one column per latent in
    LATENTS order. customer: each order's customer, as in
    TrainingSet.customer. Returns, for each name in RECOVERY_MEASURES, one
    correlation per latent: over orders, then over customer averages (each
    customer's orders averaged first, for predicted and truth alike).
    """
    p = _columns("predicted", predicted, len(LATENTS))
    t = _columns("truth", truth, len(LATENTS))
    if p.shape != t.shape:
        raise ValueError("predicted and truth must have the same shape")
    p_customers = customer_means(p, customer)
    t_customers = customer_means(t, customer)
    return {
        "pearson": pearson(p, t),
        "spearman": spearman(p, t),
        "customer_pearson": pearson(p_customers, t_customers),
        "customer_spearman": spearman(p_customers, t_customers),
    }


def kappa_summary(estimates: ArrayLike, truth: ArrayLike) -> dict[str, Array]:
    """kappa from several runs next to the truth, one value per input.

    estimates: one row per run (e.g. per seed), one column per input in
    INPUTS order. Returns truth, min, mean and max over the runs, and
    mean / truth. Raises ValueError for a wrong shape, a value that is not
    finite, or a kappa below 1.
    """
    runs = np.atleast_2d(np.asarray(estimates, dtype=np.float64))
    true = np.asarray(truth, dtype=np.float64)
    if runs.ndim != 2 or runs.shape[1] != len(INPUTS) or runs.shape[0] == 0:
        raise ValueError(f"estimates must have shape (runs, {len(INPUTS)})")
    if true.shape != (len(INPUTS),):
        raise ValueError(f"truth must have shape ({len(INPUTS)},)")
    both = np.concatenate([runs.ravel(), true])
    if not np.all(np.isfinite(both)) or np.any(both < 1.0):
        raise ValueError("every kappa must be finite and at least 1")
    mean = runs.mean(axis=0)
    return {
        "truth": true,
        "min": runs.min(axis=0),
        "mean": mean,
        "max": runs.max(axis=0),
        "mean_over_truth": mean / true,
    }
