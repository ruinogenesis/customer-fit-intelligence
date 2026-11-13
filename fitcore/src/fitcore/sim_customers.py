"""One row per simulated customer: the averages a customer-level check needs.

The SEM check (T107) fits the doc's structure per customer, because survey
answers are sparse per order and behaviours are per customer anyway. Each
customer's row holds:

- x1..x8: their average gap per input;
- item_love, item_respect, item_trust, item_convenience and nps: their
  average answer to each survey question, NaN if they never answered it;
- reorder: the share of their known reorder_14d labels that are True;
  stayed: 1 minus the share of their known churn_30d labels that are True
  (NaN if no label is known);
- their behaviours, with checkout time flipped into quick_checkout
  (minus the log of the seconds), so that more always means better;
- true_love and so on: their average true latents, for checking results.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fitcore.features import INPUTS
from fitcore.sim import Array
from fitcore.sim_behaviour import BEHAVIOURS, Behaviour
from fitcore.sim_orders import Orders
from fitcore.sim_survey import QUESTIONS


def _mean_by(
    row: NDArray[np.intp], values: Array, keep: NDArray[np.bool_], rows: int
) -> Array:
    """The mean of values for each row, over the orders kept; NaN if none."""
    total = np.bincount(row[keep], weights=values[keep], minlength=rows)
    count = np.bincount(row[keep], minlength=rows)
    out = np.full(rows, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _floats(values: tuple[float | int | bool | None, ...]) -> Array:
    """A tuple with Nones as a float array with NaNs."""
    return np.array([math.nan if v is None else float(v) for v in values])


def customer_table(orders: Orders, behaviour: Behaviour) -> dict[str, NDArray[Any]]:
    """One row per customer, in behaviour's order: averages, answers, rates.

    Raises ValueError unless behaviour covers exactly the customers in
    orders.
    """
    ids = behaviour.customer_ids
    if set(ids) != set(orders.customer_ids) or len(set(ids)) != len(ids):
        raise ValueError("behaviour must cover exactly the customers in orders")
    position = {customer: i for i, customer in enumerate(ids)}
    row = np.array([position[c] for c in orders.customer_ids], dtype=np.intp)
    rows = len(ids)
    every = np.ones(len(orders), dtype=bool)

    table: dict[str, NDArray[Any]] = {
        "customer_id": np.array(ids),
        "orders": np.bincount(row, minlength=rows),
    }
    for i, key in enumerate(INPUTS):
        table[key] = _mean_by(row, orders.gaps[:, i], every, rows)

    asked = np.array([q or "" for q in orders.survey_items])
    answers = _floats(orders.survey_answers)
    for question in QUESTIONS:
        name = "nps" if question == "nps" else f"item_{question}"
        table[name] = _mean_by(row, answers, asked == question, rows)

    reorder = _floats(orders.reorder_14d)
    churn = _floats(orders.churn_30d)
    table["reorder"] = _mean_by(row, reorder, ~np.isnan(reorder), rows)
    table["stayed"] = 1.0 - _mean_by(row, churn, ~np.isnan(churn), rows)

    for name in BEHAVIOURS:
        if name == "checkout_seconds":
            table["quick_checkout"] = -np.log(behaviour.values[name])
        else:
            table[name] = np.array(behaviour.values[name])

    for latent, values in behaviour.true_means.items():
        table[f"true_{latent}"] = np.array(values)
    return table
