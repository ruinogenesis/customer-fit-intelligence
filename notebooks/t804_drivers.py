# %% [markdown]
# # T804: does /drivers find the planted trouble?
#
# The simulator can plant trouble (T315): with plant=True, 80% of late
# orders come from rest-00 or rest-01, 80% of cold food from rest-02, and
# 80% of rude couriers are courier-00, -01 or -02. Here 20 populations
# (seeds 2026 to 2045) of four sizes (50, 100, 200 and 400 customers over
# 90 days) are replayed event by event through fitcore.scoring with the
# T112 model, as the API scores them. Each order keeps its latest scoring's
# contributions to fit, and the groups are ranked as GET /drivers ranks
# them (cfi_api.drivers): each input's average contribution per restaurant
# or courier (drag), minus its average over every order (excess); groups
# with at least 5 orders, worst first by their most negative excess.
#
# Two plainer rankings for comparison: lowest average fit first, and each
# group's most negative drag without subtracting the overall average.
#
# For each: how many of the top three groups are planted (precision at 3),
# the planted groups' ranks, and whether each planted group's first driver
# is its planted input (late: x1, cold food: x6, rude courier: x5). And for
# every scoring in every replay, that each of its six scores' contributions
# add up to its change from an order exactly as expected (within 1e-12);
# the run stops if one does not. Saves notebooks/results/t804_drivers.csv.
# About eight minutes on one CPU core.

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fitcore.features import INPUTS
from fitcore.model import SCORES, ModelParams
from fitcore.schemas import load_event
from fitcore.scoring import Customer, apply_event, order_id_for, score
from fitcore.sim_events import PLANTED, simulate_events

SEEDS = tuple(range(2026, 2046))
SIZES = (50, 100, 200, 400)
DAYS = 90.0
MIN_ORDERS = 5  # GET /drivers' default
TOP = 3
SPLIT_TOLERANCE = 1e-12  # how far a score's contributions may miss its change
INCIDENT_INPUT = {"late": "x1", "cold_food": "x6", "rude_courier": "x5"}


def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


RESULTS = repo_root() / "notebooks" / "results"
params = ModelParams.from_json((RESULTS / "t112_model.json").read_bytes())
as_expected = score(params, {})  # every gap 0: what each split starts from

# Which input each planted group should show, by grouping.
EXPECTED: dict[str, dict[str, str]] = {"restaurant": {}, "courier": {}}
for incident, (field, names) in PLANTED.items():
    by = "restaurant" if field == "restaurant_id" else "courier"
    for name in names:
        EXPECTED[by][name] = INCIDENT_INPUT[incident]


def split_error(scored, contributions: dict[str, dict[str, float]]) -> float:
    """The most any of the six scores' contributions miss its change."""
    return max(
        abs(
            sum(contributions[name].values())
            - (getattr(scored, name) - getattr(as_expected, name))
        )
        for name in SCORES
    )


def scored_orders(events: list[dict]) -> tuple[pd.DataFrame, int, float]:
    """Every scored order's latest contributions to fit, with its groups; how
    many scorings there were, and the largest split error among them."""
    customers: dict[str, Customer] = {}
    orders: dict = {}
    latest: dict[str, dict[str, float]] = {}
    fit: dict[str, float] = {}
    scorings, worst = 0, 0.0
    for raw in events:
        event = load_event(raw)
        customer = customers.get(event["customer_id"]) or Customer.new(
            event["customer_id"]
        )
        wanted = order_id_for(event, customer)
        applied = apply_event(
            params, event, customer, None if wanted is None else orders.get(wanted)
        )
        customers[event["customer_id"]] = applied.customer
        if applied.order is not None:
            orders[applied.order.order_id] = applied.order
        if applied.explained is not None and applied.scored is not None:
            assert applied.order is not None
            latest[applied.order.order_id] = applied.explained.contributions["fit"]
            fit[applied.order.order_id] = applied.scored.fit
            scorings += 1
            worst = max(
                worst, split_error(applied.scored, applied.explained.contributions)
            )
    table = pd.DataFrame(
        [
            {
                "order_id": order_id,
                "restaurant": orders[order_id].restaurant_id,
                "courier": orders[order_id].courier_id,
                "fit": fit[order_id],
                **contributions,
            }
            for order_id, contributions in latest.items()
        ]
    )
    return table, scorings, worst


def rankings(table: pd.DataFrame, by: str) -> pd.DataFrame:
    """Each group's drag, excess and rank, the way GET /drivers works them out."""
    inputs = list(INPUTS)
    overall = table[inputs].mean()  # every order, before the min_orders filter
    groups = table.groupby(by).agg(orders=("fit", "size"), fit=("fit", "mean"))
    drag = table.groupby(by)[inputs].mean()
    excess = drag - overall
    kept = groups["orders"] >= MIN_ORDERS
    out = groups[kept].copy()
    out["worst_excess"] = excess[kept].min(axis=1)
    out["first_driver"] = excess[kept].idxmin(axis=1)
    out["worst_drag"] = drag[kept].min(axis=1)
    out["first_drag"] = drag[kept].idxmin(axis=1)
    ids = out.index.to_series()
    # worst first, ties by id, as the API sorts
    out["rank_excess"] = _rank(out["worst_excess"], ids)
    out["rank_fit"] = _rank(out["fit"], ids)
    out["rank_drag"] = _rank(out["worst_drag"], ids)
    return out


def _rank(values: pd.Series, ids: pd.Series) -> pd.Series:
    order = sorted(values.index, key=lambda g: (values[g], ids[g]))
    return pd.Series({g: i + 1 for i, g in enumerate(order)})


# %% Replay every population
rows: list[dict] = []
checked, largest = 0, 0.0
started = time.perf_counter()
for size in SIZES:
    for seed in SEEDS:
        table, scorings, worst = scored_orders(
            simulate_events(size, seed, DAYS, plant=True)
        )
        if worst > SPLIT_TOLERANCE:
            raise AssertionError(
                f"a split misses its score by {worst:g}: {size}, {seed}"
            )
        checked, largest = checked + scorings, max(largest, worst)
        rows.append(
            {
                "size": size,
                "seed": seed,
                "orders": len(table),
                "by": None,
                "method": None,
                "measure": "scorings with exact splits",
                "group": None,
                "value": float(scorings),
            }
        )
        for by, expected in EXPECTED.items():
            ranked = rankings(table, by)
            rows.append(
                {
                    "size": size,
                    "seed": seed,
                    "orders": len(table),
                    "by": by,
                    "method": None,
                    "measure": "groups ranked",
                    "group": None,
                    "value": float(len(ranked)),
                }
            )
            for method in ("excess", "fit", "drag"):
                top = set(ranked.sort_values(f"rank_{method}").index[:TOP])
                rows.append(
                    {
                        "size": size,
                        "seed": seed,
                        "orders": len(table),
                        "by": by,
                        "method": method,
                        "measure": f"precision at {TOP}",
                        "group": None,
                        "value": len(top & set(expected)) / TOP,
                    }
                )
            for group, planted_input in expected.items():
                if group not in ranked.index:  # too few orders to rank
                    continue
                g = ranked.loc[group]
                for method in ("excess", "fit", "drag"):
                    rows.append(
                        {
                            "size": size,
                            "seed": seed,
                            "orders": len(table),
                            "by": by,
                            "method": method,
                            "measure": "rank",
                            "group": group,
                            "value": float(g[f"rank_{method}"]),
                        }
                    )
                for method, column in (
                    ("excess", "first_driver"),
                    ("drag", "first_drag"),
                ):
                    rows.append(
                        {
                            "size": size,
                            "seed": seed,
                            "orders": len(table),
                            "by": by,
                            "method": method,
                            "measure": "right first driver",
                            "group": group,
                            "value": float(g[column] == planted_input),
                        }
                    )
    print(f"{size} customers done, {time.perf_counter() - started:.0f} s so far")

# %% Save
RESULTS.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(rows)
results.to_csv(RESULTS / "t804_drivers.csv", index=False, float_format="%.6f")

# %% Summary: mean over the 20 populations
pd.set_option("display.width", 160)
print(results.groupby("size")["orders"].mean().round(0).rename("orders per population"))
print(
    results.pivot_table(
        index=["by", "measure", "method"],
        columns="size",
        values="value",
        aggfunc="mean",
    ).round(3)
)
print(
    results[results["measure"] == "rank"]
    .pivot_table(
        index=["by", "group", "method"],
        columns="size",
        values="value",
        aggfunc="median",
    )
    .round(1)
)
print(
    f"{checked:,} scorings, six scores each: every split adds up "
    f"(largest miss {largest:.1e})"
)
print(np.round(time.perf_counter() - started))
