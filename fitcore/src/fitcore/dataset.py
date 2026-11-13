"""Training rows for the v2 model: one row per order, behaviours per customer.

The model scores orders, so it learns from orders. Each row holds one order's
gaps (x1..x8, own units), the customer who placed it, its outcome labels and
the answer to its survey question, if it had one. Behaviours describe a
customer, not an order, so they sit in a table of their own, one row per
customer.

Anything that is not known is NaN: a label whose window runs past the data,
a survey question that was not asked. The losses in fitcore.train skip NaNs,
so a missing value never counts as an answer.

numpy only: the simulator builds a TrainingSet now, and the BigQuery export
will build the same thing later.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from fitcore.features import INPUTS
from fitcore.model import LATENTS
from fitcore.sim import Array
from fitcore.sim_behaviour import Behaviour
from fitcore.sim_orders import Orders

# The latents that have a survey item, in the column order of TrainingSet.items.
ITEM_LATENTS: tuple[str, ...] = LATENTS[:4]

# Each behaviour column and the latent it anchors, in the column order of
# TrainingSet.behaviours. Favorites are a count and checkout is a time, so
# both are put on a log scale; quick_checkout is minus the log of the seconds,
# so that more always means better.
ANCHOR_BEHAVIOURS: Mapping[str, str] = MappingProxyType(
    {
        "log_favorites": "love",
        "promo_free_share": "love",
        "average_rating": "respect",
        "new_restaurant_share": "trust",
        "saved_payment": "trust",
        "quick_checkout": "convenience",
    }
)


def _read_only(
    name: str, value: ArrayLike, shape: tuple[int, ...], dtype: DTypeLike
) -> NDArray[np.generic]:
    """A read-only copy of value, checked for shape."""
    array = np.array(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


def _check_answers(name: str, values: Array, low: int, high: int) -> None:
    """Every value that is not NaN must be a whole number from low to high."""
    known = values[~np.isnan(values)]
    if not np.all((known == np.round(known)) & (known >= low) & (known <= high)):
        raise ValueError(f"{name} must be NaN or a whole number from {low} to {high}")


@dataclass(frozen=True, eq=False)
class TrainingSet:
    """Orders to train on, one row each, and their customers' behaviours.

    customer: each order's customer, as a position in customer_ids.
    gaps: n x 8, own units. reorder, churn: 1.0, 0.0 or NaN. nps: 0 to 10 or
    NaN. items: n x 4, the 1-to-7 answer for each latent in ITEM_LATENTS, NaN
    where that question was not asked. behaviours: one row per customer, one
    column per ANCHOR_BEHAVIOURS. truth: the simulator's true latents and fit
    per order (n x 5, in LATENTS order), or None for real data.

    Arrays are copied and made read-only. Raises ValueError for a wrong
    shape, a customer with no orders, a gap, behaviour or truth that is not
    finite, or a label or answer out of range.
    """

    customer_ids: tuple[str, ...]
    customer: NDArray[np.intp]
    gaps: Array
    reorder: Array
    churn: Array
    nps: Array
    items: Array
    behaviours: Array
    truth: Array | None = None

    def __post_init__(self) -> None:
        n = len(np.asarray(self.customer))
        customers = len(self.customer_ids)
        shapes: dict[str, tuple[int, ...]] = {
            "gaps": (n, len(INPUTS)),
            "reorder": (n,),
            "churn": (n,),
            "nps": (n,),
            "items": (n, len(ITEM_LATENTS)),
            "behaviours": (customers, len(ANCHOR_BEHAVIOURS)),
        }
        if self.truth is not None:
            shapes["truth"] = (n, len(LATENTS))
        object.__setattr__(
            self, "customer", _read_only("customer", self.customer, (n,), np.intp)
        )
        for name, shape in shapes.items():
            value = _read_only(name, getattr(self, name), shape, np.float64)
            object.__setattr__(self, name, value)

        if np.any((self.customer < 0) | (self.customer >= customers)):
            raise ValueError("customer must be a position in customer_ids")
        if np.bincount(self.customer, minlength=customers).min(initial=1) < 1:
            raise ValueError("every customer must have at least one order")
        for name in ("gaps", "behaviours", "truth"):
            value = getattr(self, name)
            if value is not None and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        _check_answers("reorder", self.reorder, 0, 1)
        _check_answers("churn", self.churn, 0, 1)
        _check_answers("nps", self.nps, 0, 10)
        _check_answers("items", self.items, 1, 7)

    def __len__(self) -> int:
        return len(self.customer)

    def select(self, keep: ArrayLike) -> "TrainingSet":
        """Only the customers where keep is True, with all of their orders.

        keep has one True or False per customer. Raises ValueError for a
        wrong shape.
        """
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != (len(self.customer_ids),):
            raise ValueError(f"keep must have shape ({len(self.customer_ids)},)")
        rows = mask[self.customer]
        new_position = np.cumsum(mask) - 1
        return TrainingSet(
            customer_ids=tuple(
                c for c, k in zip(self.customer_ids, mask, strict=True) if k
            ),
            customer=new_position[self.customer[rows]],
            gaps=self.gaps[rows],
            reorder=self.reorder[rows],
            churn=self.churn[rows],
            nps=self.nps[rows],
            items=self.items[rows],
            behaviours=self.behaviours[mask],
            truth=None if self.truth is None else self.truth[rows],
        )


def gap_scale(gaps: ArrayLike) -> Array:
    """Each input's typical gap: the root-mean-square of its column.

    No centring, so 0 still means "as expected". The network divides each
    transformed gap by this. Compute it on the training set and keep it with
    the model. Raises ValueError unless gaps is (n, 8) with n of 1 or more,
    and finite, or if every gap of some input is 0.
    """
    g = np.asarray(gaps, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != len(INPUTS) or g.shape[0] == 0:
        raise ValueError(f"gaps must have shape (n, {len(INPUTS)}), got {g.shape}")
    if not np.all(np.isfinite(g)):
        raise ValueError("gaps must be finite")
    scale = np.sqrt(np.mean(g**2, axis=0))
    if np.any(scale == 0.0):
        raise ValueError("every input needs at least one gap that is not 0")
    return scale


def split_by_customer(
    data: TrainingSet, test_share: float, seed: int
) -> tuple[TrainingSet, TrainingSet]:
    """(train, test): every customer, with all their orders, on one side.

    A customer's orders share their habits, so holding out orders one by one
    would leak them into training. round(test_share * customers) customers,
    picked at random by seed, go to test. Raises ValueError unless both
    sides get at least one customer.
    """
    count = len(data.customer_ids)
    n_test = round(test_share * count)
    if not 0 < n_test < count:
        raise ValueError("both sides of the split need at least one customer")
    test = np.zeros(count, dtype=bool)
    test[np.random.default_rng(seed).permutation(count)[:n_test]] = True
    return data.select(~test), data.select(test)


def _floats(values: tuple[float | int | bool | None, ...]) -> Array:
    """A tuple with Nones as a float array with NaNs."""
    return np.array([np.nan if v is None else float(v) for v in values])


def from_simulation(orders: Orders, behaviour: Behaviour) -> TrainingSet:
    """A TrainingSet from simulated orders and behaviours, truth included.

    Customers come in behaviour's order. Raises ValueError unless behaviour
    covers exactly the customers in orders.
    """
    ids = behaviour.customer_ids
    if set(ids) != set(orders.customer_ids) or len(set(ids)) != len(ids):
        raise ValueError("behaviour must cover exactly the customers in orders")
    position = {customer: i for i, customer in enumerate(ids)}

    asked = np.array([q or "" for q in orders.survey_items])
    answers = _floats(orders.survey_answers)
    items = np.full((len(orders), len(ITEM_LATENTS)), np.nan)
    for column, latent in enumerate(ITEM_LATENTS):
        items[asked == latent, column] = answers[asked == latent]

    values = behaviour.values
    columns = {
        "log_favorites": np.log1p(values["favorites_saved"]),
        "promo_free_share": values["promo_free_share"],
        "average_rating": values["average_rating"],
        "new_restaurant_share": values["new_restaurant_share"],
        "saved_payment": values["saved_payment"],
        "quick_checkout": -np.log(values["checkout_seconds"]),
    }
    truth = [orders.latents[name] for name in ITEM_LATENTS] + [orders.fit]
    return TrainingSet(
        customer_ids=ids,
        customer=np.array([position[c] for c in orders.customer_ids]),
        gaps=orders.gaps,
        reorder=_floats(orders.reorder_14d),
        churn=_floats(orders.churn_30d),
        nps=_floats(orders.nps),
        items=items,
        behaviours=np.column_stack([columns[name] for name in ANCHOR_BEHAVIOURS]),
        truth=np.column_stack(truth),
    )
