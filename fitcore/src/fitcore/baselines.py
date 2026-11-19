"""What the paper compares CFI with, and how it scores them (T802).

The paper judges CFI against models that see less (an order's history
alone) or explain less (an unconstrained network on the same gaps). This
module holds the parts worth testing; the notebook fits the scikit-learn
models around them:

- history(): what each order's history says at the moment it is placed:
  its number, the days since the customer's first and previous orders, and
  T113's order rate (orders per 30 days, with a 7-day head start);
- BG/NBD (Fader, Hardie and Lee 2005): repeat buying from frequency and
  recency alone. fit_bgnbd() finds its four parameters by maximum
  likelihood; p_no_order() gives the chance of no order in the next tau
  days, which is churn_30d's question (tau 30) and 1 minus reorder_14d's
  (tau 14);
- auc(), log_loss(), brier() and ece(): the scores, written out so that
  they can be checked against their definitions;
- event_mask() and carry_forward(): which inputs v1 events measure on a
  simulated order (fitcore.sim_events' rules), and the two ways to fill
  the inputs an order did not measure.

numpy, plus scipy for the BG/NBD fit: fitcore's `paper` extra.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fitcore.features import INPUTS

Array = NDArray[np.float64]

# T113's order rate: orders so far per 30 days, after a 7-day head start,
# so that a first order reads as about weekly.
HEAD_START_DAYS = 7.0
RATE_DAYS = 30.0

# What v1 events measure (fitcore.sim_events): x1..x3 on every order; the
# courier and food scores (x5, x6) when the order is rated, which every
# order with an incident is; support-case satisfaction (x8) on orders with
# an incident; x4 and x7 never (T209).
ALWAYS = ("x1", "x2", "x3")
RATED = ("x5", "x6")
SUPPORT = ("x8",)


@dataclass(frozen=True)
class History:
    """Per order, in the rows' order: what its history says when placed."""

    number: Array  # 1 for a customer's first order
    days_since_first: Array
    days_since_previous: Array  # NaN on a first order
    log_rate: Array


def history(customers: Sequence[str], placed_at: Sequence[datetime]) -> History:
    """Each order's history, from only the orders placed before it.

    Rows may come in any order; a customer's orders are ranked by time.
    Raises ValueError if the two sequences differ in length.
    """
    if len(customers) != len(placed_at):
        raise ValueError("customers and placed_at must have the same length")
    n = len(customers)
    days = np.array([t.timestamp() / 86_400.0 for t in placed_at], dtype=np.float64)
    number = np.zeros(n)
    first = np.zeros(n)
    previous = np.full(n, np.nan)
    rows_of: dict[str, list[int]] = {}
    for row, customer in enumerate(customers):
        rows_of.setdefault(customer, []).append(row)
    for rows in rows_of.values():
        ranked = sorted(rows, key=lambda r: (days[r], r))
        start = days[ranked[0]]
        for k, row in enumerate(ranked):
            number[row] = k + 1
            first[row] = days[row] - start
            if k > 0:
                previous[row] = days[row] - days[ranked[k - 1]]
    log_rate = np.log(number / (first + HEAD_START_DAYS) * RATE_DAYS)
    return History(number, first, previous, log_rate)


# --- BG/NBD ---


# Where fit_bgnbd searches: each parameter between e^-8 and e^8.
LOG_LOW, LOG_HIGH = -8.0, 8.0


@dataclass(frozen=True)
class BgNbd:
    """BG/NBD's parameters: purchase rates ~ Gamma(r, alpha), dropout ~ Beta(a, b)."""

    r: float
    alpha: float
    a: float
    b: float


def _gammaln(x: ArrayLike) -> Array:
    from scipy.special import gammaln

    return np.asarray(gammaln(x), dtype=np.float64)


def _betaln(x: ArrayLike, y: ArrayLike) -> Array:
    from scipy.special import betaln

    return np.asarray(betaln(x, y), dtype=np.float64)


def bgnbd_log_likelihood(
    params: BgNbd, x: ArrayLike, t_x: ArrayLike, T: ArrayLike
) -> Array:
    """Each customer's log-likelihood (Fader, Hardie and Lee 2005, eq. 6).

    x: repeat purchases (orders after the first); t_x: days from the first
    order to the last; T: days from the first order to the end of the data.
    """
    x, t_x, T = (np.asarray(v, dtype=np.float64) for v in (x, t_x, T))
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    common = _gammaln(r + x) - _gammaln(r) + r * np.log(alpha)
    alive = _betaln(a, b + x) - _betaln(a, b) + common - (r + x) * np.log(alpha + T)
    gone = np.full_like(x, -np.inf)
    repeat = x > 0
    gone[repeat] = (
        _betaln(a + 1, b + x[repeat] - 1)
        - _betaln(a, b)
        + common[repeat]
        - (r + x[repeat]) * np.log(alpha + t_x[repeat])
    )
    return np.asarray(np.logaddexp(alive, gone), dtype=np.float64)


def fit_bgnbd(x: ArrayLike, t_x: ArrayLike, T: ArrayLike) -> BgNbd:
    """BG/NBD's maximum-likelihood parameters for these customers.

    Optimises the log of each parameter, within [LOG_LOW, LOG_HIGH], so all
    four stay positive. Raises ValueError if the search fails.
    """
    from scipy.optimize import minimize

    x, t_x, T = (np.asarray(v, dtype=np.float64) for v in (x, t_x, T))

    def loss(logs: Array) -> float:
        r, alpha, a, b = np.exp(logs)
        return float(-bgnbd_log_likelihood(BgNbd(r, alpha, a, b), x, t_x, T).sum())

    # Bounded: when few customers drop out, a and b can grow together
    # without changing the likelihood much, and an unbounded search drifts.
    result = minimize(
        loss,
        np.log([1.0, 10.0, 1.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(LOG_LOW, LOG_HIGH)] * 4,
    )
    if not result.success:
        raise ValueError(f"BG/NBD fit failed: {result.message}")
    r, alpha, a, b = (float(v) for v in np.exp(result.x))
    return BgNbd(r, alpha, a, b)


def p_alive(params: BgNbd, x: ArrayLike, t_x: ArrayLike, T: ArrayLike) -> Array:
    """The chance each customer has not dropped out (eq. 13)."""
    x, t_x, T = (np.asarray(v, dtype=np.float64) for v in (x, t_x, T))
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    odds = np.zeros_like(x)
    repeat = x > 0
    odds[repeat] = (a / (b + x[repeat] - 1)) * (
        (alpha + T[repeat]) / (alpha + t_x[repeat])
    ) ** (r + x[repeat])
    return 1.0 / (1.0 + odds)


def p_no_order(
    params: BgNbd, x: ArrayLike, t_x: ArrayLike, T: ArrayLike, tau: float
) -> Array:
    """The chance of no order in the next tau days.

    Dropped out, or alive with no purchase in tau days: a live customer's
    rate has the posterior Gamma(r + x, alpha + T), so that chance is
    ((alpha + T) / (alpha + T + tau)) ** (r + x).
    """
    x, t_x, T = (np.asarray(v, dtype=np.float64) for v in (x, t_x, T))
    alive = p_alive(params, x, t_x, T)
    quiet = ((params.alpha + T) / (params.alpha + T + tau)) ** (params.r + x)
    return 1.0 - alive + alive * quiet


# --- Scores ---


def _known(p: ArrayLike, y: ArrayLike) -> tuple[Array, Array]:
    """p and y where y is known (not NaN), checked to match in length."""
    p_, y_ = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if p_.shape != y_.shape:
        raise ValueError("p and y must have the same shape")
    keep = ~np.isnan(y_)
    return p_[keep], y_[keep]


def auc(p: ArrayLike, y: ArrayLike) -> float:
    """The chance a random positive outranks a random negative, ties half.

    Rows whose y is NaN are left out. Raises ValueError unless both classes
    are present.
    """
    p_, y_ = _known(p, y)
    positives = y_ == 1.0
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc needs both classes")
    order = np.argsort(p_, kind="mergesort")
    ranks = np.empty(len(p_))
    sorted_p = p_[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss(p: ArrayLike, y: ArrayLike, eps: float = 1e-15) -> float:
    """Mean cross-entropy, with p clipped to [eps, 1 - eps]."""
    p_, y_ = _known(p, y)
    q = np.clip(p_, eps, 1.0 - eps)
    return float(-np.mean(y_ * np.log(q) + (1.0 - y_) * np.log(1.0 - q)))


def brier(p: ArrayLike, y: ArrayLike) -> float:
    """Mean squared difference between the chance and the outcome."""
    p_, y_ = _known(p, y)
    return float(np.mean((p_ - y_) ** 2))


def ece(p: ArrayLike, y: ArrayLike, bins: int = 10) -> float:
    """Expected calibration error: equal-width bins of p, each weighted by its share."""
    p_, y_ = _known(p, y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p_, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for k in range(bins):
        inside = which == k
        if inside.any():
            total += inside.mean() * abs(p_[inside].mean() - y_[inside].mean())
    return float(total)


# --- What events measure ---


def event_mask(
    incidents: Sequence[str | None], rating_rate: float, seed: int
) -> NDArray[np.bool_]:
    """n x 8: which inputs v1 events measure on each simulated order.

    An order is rated when it had an incident, or else with chance
    rating_rate (one draw per order from seed).
    """
    n = len(incidents)
    had = np.array([i is not None for i in incidents])
    rated = had | (np.random.default_rng(seed).random(n) < rating_rate)
    mask = np.zeros((n, len(INPUTS)), dtype=bool)
    for key in ALWAYS:
        mask[:, INPUTS.index(key)] = True
    for key in RATED:
        mask[:, INPUTS.index(key)] = rated
    for key in SUPPORT:
        mask[:, INPUTS.index(key)] = had
    return mask


def carry_forward(
    gaps: ArrayLike, mask: ArrayLike, customers: Sequence[str], order: ArrayLike
) -> Array:
    """Gaps with each unmeasured input replaced by the customer's last measured one.

    order ranks each customer's rows in time (e.g. the order number). An
    input never measured so far stays 0, "as expected".
    """
    g = np.asarray(gaps, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if g.shape != m.shape or g.shape[0] != len(customers):
        raise ValueError("gaps, mask and customers must describe the same rows")
    ranks = np.asarray(order, dtype=np.float64)
    out = np.where(m, g, 0.0)
    rows_of: dict[str, list[int]] = {}
    for row, customer in enumerate(customers):
        rows_of.setdefault(customer, []).append(row)
    for rows in rows_of.values():
        last = np.zeros(g.shape[1])
        for row in sorted(rows, key=lambda r: (ranks[r], r)):
            last = np.where(m[row], g[row], last)
            out[row] = last
    return out
