"""Real orders: the Olist public dataset (T805).

CFI's simulation study cannot show that real customers judge orders by gaps
and weigh shortfalls more than surpluses. Olist's public dataset of about
100,000 Brazilian e-commerce orders (2016 to 2018, CC BY-NC-SA 4.0, on
Kaggle) can test both on real people: every order has a promised delivery
date, an actual one, a price, a freight charge, and usually a 1-to-5 review.

- load(): one row per delivered order, from four of the dataset's files:
  the gap in days (promised minus actual, positive when early, x1's sign),
  the delivery time itself, the fee load (freight over price, x3's
  measure), the review, the order's latest if it has several, and when
  that review was answered;
- fit_ordinal(): a cumulative-logit model of an answer on 0..K-1 (here the
  review minus 1), P(answer <= k) = sigmoid(cut_k - x @ beta), the form CFI
  uses for NPS and the survey items;
- asymmetric() and profile_kappa(): CFI's transform g, and the likelihood of
  each kappa with the rest refitted, for a kappa and its confidence interval.

numpy, plus scipy and pandas: fitcore's `paper` extra. The files stay out of
the repo (notebooks/data/ is ignored); see notebooks/t805_olist.py.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}

DAY_SECONDS = 86_400.0

# A 95% likelihood-ratio interval: twice the log-likelihood falls by at most
# the chi-squared(1) quantile 3.841, so the negative log-likelihood rises by
# at most half of it.
LR_95 = 3.841458820694124 / 2.0


def load(folder: Path) -> Any:
    """One row per delivered order with a delivery date, as a pandas DataFrame.

    Columns: order_id, customer (customer_unique_id, the same person across
    orders), purchased, delivered, promised, gap_days (promised minus
    delivered, in days; positive = early), days (purchase to delivery),
    price and freight (sums over the order's items), fee (freight / price),
    review (1 to 5, the order's latest review; NaN if none) and reviewed
    (when that review was answered; NaT if none or not recorded). Olist
    sends its survey on the promised date when the order has not arrived,
    so a late order is often reviewed before it is delivered (T810). Raises
    FileNotFoundError naming any missing file.
    """
    import pandas as pd

    missing = [name for name in FILES.values() if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing from {folder}: {', '.join(missing)}")
    times = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = pd.read_csv(folder / FILES["orders"])
    for column in times:
        orders[column] = pd.to_datetime(orders[column], format="ISO8601")
    items = pd.read_csv(folder / FILES["items"])
    reviews = pd.read_csv(folder / FILES["reviews"])
    for column in ("review_creation_date", "review_answer_timestamp"):
        reviews[column] = pd.to_datetime(reviews[column], format="ISO8601")
    customers = pd.read_csv(folder / FILES["customers"])

    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
    ]
    latest = (
        reviews.sort_values(["review_creation_date", "review_id"])
        .drop_duplicates("order_id", keep="last")
        .set_index("order_id")
    )
    money = items.groupby("order_id").agg(
        price=("price", "sum"), freight=("freight_value", "sum")
    )
    people = customers.set_index("customer_id")["customer_unique_id"]

    out = pd.DataFrame(
        {
            "order_id": delivered["order_id"].to_numpy(),
            "customer": delivered["customer_id"].map(people).to_numpy(),
            "purchased": delivered["order_purchase_timestamp"].to_numpy(),
            "delivered": delivered["order_delivered_customer_date"].to_numpy(),
            "promised": delivered["order_estimated_delivery_date"].to_numpy(),
        }
    )
    out["gap_days"] = (
        out["promised"] - out["delivered"]
    ).dt.total_seconds() / DAY_SECONDS
    out["days"] = (out["delivered"] - out["purchased"]).dt.total_seconds() / DAY_SECONDS
    out = out.join(money, on="order_id")
    out["fee"] = out["freight"] / out["price"]
    out["review"] = out["order_id"].map(latest["review_score"]).astype(float)
    out["reviewed"] = out["order_id"].map(latest["review_answer_timestamp"])
    return out.sort_values(["purchased", "order_id"]).reset_index(drop=True)


def asymmetric(gaps: ArrayLike, kappa: float) -> Array:
    """CFI's g: a gap as it is when 0 or more, times kappa when below 0."""
    g = np.asarray(gaps, dtype=np.float64)
    return np.where(g >= 0.0, g, kappa * g)


@dataclass(frozen=True)
class OrdinalFit:
    """A cumulative-logit fit: P(answer <= k) = sigmoid(cuts[k] - x @ beta)."""

    beta: Array
    cuts: Array
    nll: float  # summed over the rows


def _cuts(free: Array) -> Array:
    """Increasing cuts from the first cut and the logs of the steps between."""
    return np.concatenate([free[:1], free[0] + np.cumsum(np.exp(free[1:]))])


def ordinal_nll(beta: ArrayLike, cuts: ArrayLike, x: ArrayLike, y: ArrayLike) -> Array:
    """Each row's negative log-likelihood; y holds answers 0..len(cuts)."""
    from scipy.special import expit

    eta = np.asarray(x, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)
    edges = np.concatenate([[-np.inf], np.asarray(cuts, dtype=np.float64), [np.inf]])
    k = np.asarray(y, dtype=np.intp)
    upper = expit(edges[k + 1] - eta)
    lower = expit(edges[k] - eta)
    return np.asarray(-np.log(np.clip(upper - lower, 1e-300, None)), dtype=np.float64)


def ordinal_gradient(
    beta: ArrayLike, cuts: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[Array, Array]:
    """The gradient of the summed ordinal_nll: (by beta, by cuts)."""
    from scipy.special import expit

    x_ = np.asarray(x, dtype=np.float64)
    c = np.asarray(cuts, dtype=np.float64)
    eta = x_ @ np.asarray(beta, dtype=np.float64)
    edges = np.concatenate([[-np.inf], c, [np.inf]])
    k = np.asarray(y, dtype=np.intp)
    upper = expit(edges[k + 1] - eta)
    lower = expit(edges[k] - eta)
    chance = np.clip(upper - lower, 1e-300, None)
    # sigmoid'(t) = s(1 - s), which is 0 at the infinite edges
    d_upper = upper * (1.0 - upper) / chance
    d_lower = lower * (1.0 - lower) / chance
    by_eta = d_upper - d_lower
    by_cuts = np.zeros(len(c))
    has_upper, has_lower = k < len(c), k > 0
    np.add.at(by_cuts, k[has_upper], -d_upper[has_upper])
    np.add.at(by_cuts, k[has_lower] - 1, d_lower[has_lower])
    return np.asarray(x_.T @ by_eta, dtype=np.float64), by_cuts


def fit_ordinal(x: ArrayLike, y: ArrayLike, classes: int) -> OrdinalFit:
    """The maximum-likelihood cumulative logit of y (0..classes-1) on x.

    x: n x p, no intercept column (the cuts are the intercepts); p may be 0.
    Raises ValueError if an answer is out of range or the search fails.
    """
    from scipy.optimize import minimize

    x_ = np.asarray(x, dtype=np.float64).reshape(len(np.asarray(y)), -1)
    y_ = np.asarray(y, dtype=np.intp)
    if y_.min() < 0 or y_.max() >= classes:
        raise ValueError(f"answers must run from 0 to {classes - 1}")
    p = x_.shape[1]
    shares = np.clip(
        np.cumsum(np.bincount(y_, minlength=classes))[:-1] / len(y_), 1e-6, 1 - 1e-6
    )
    start_cuts = np.log(shares / (1.0 - shares))
    start = np.concatenate(
        [np.zeros(p), start_cuts[:1], np.log(np.maximum(np.diff(start_cuts), 1e-3))]
    )

    def loss(theta: Array) -> tuple[float, Array]:
        free = theta[p:]
        cuts = _cuts(free)
        nll = float(ordinal_nll(theta[:p], cuts, x_, y_).sum())
        by_beta, by_cuts = ordinal_gradient(theta[:p], cuts, x_, y_)
        # cuts[j] = free[0] + sum of exp(free[1..j]): back to the free numbers
        tail = np.cumsum(by_cuts[::-1])[::-1]
        by_free = np.concatenate([[tail[0]], np.exp(free[1:]) * tail[1:]])
        return nll, np.concatenate([by_beta, by_free])

    # An exact gradient: with finite differences the search stops early on
    # sums this large, and a profile over kappa comes out jagged.
    result = minimize(loss, start, jac=True, method="BFGS", options={"gtol": 1e-6})
    if not np.isfinite(result.fun):
        raise ValueError(f"ordinal fit failed: {result.message}")
    return OrdinalFit(result.x[:p].copy(), _cuts(result.x[p:]), float(result.fun))


def profile_kappa(
    gaps: ArrayLike,
    scale: float,
    others: ArrayLike,
    y: ArrayLike,
    classes: int,
    kappas: Sequence[float],
) -> list[tuple[float, float]]:
    """(kappa, summed negative log-likelihood) for each kappa, everything
    else refitted: the gap enters as asymmetric(gap, kappa) / scale, beside
    the columns of others (n x q, q may be 0)."""
    y_ = np.asarray(y)
    rest = np.asarray(others, dtype=np.float64).reshape(len(y_), -1)
    out = []
    for kappa in kappas:
        x = np.column_stack([asymmetric(gaps, kappa) / scale, rest])
        out.append((float(kappa), fit_ordinal(x, y_, classes).nll))
    return out


def kappa_interval(
    profile: Sequence[tuple[float, float]],
) -> tuple[float, float, float]:
    """(best kappa, low, high): the grid's best, and the outermost kappas whose
    negative log-likelihood is within LR_95 of it (a 95% interval, as fine as
    the grid). A bound at the grid's edge means the interval reaches past it."""
    kappas = np.array([k for k, _ in profile])
    nll = np.array([v for _, v in profile])
    best = int(np.argmin(nll))
    inside = kappas[nll <= nll[best] + LR_95]
    return float(kappas[best]), float(inside.min()), float(inside.max())
