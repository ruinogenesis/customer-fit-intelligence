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
# the run stops if one does not.
#
# T811 adds what the third review asked for, decided before the run:
#
# - Rankings with no model at all, each from one raw measure: couriers by
#   their mean courier rating (lowest first), restaurants by their mean
#   minutes late against the promise (highest first) and by their mean food
#   rating (lowest first). A group needs at least 5 orders, as above, and at
#   least one such measure. Each is asked for as many groups as are planted
#   for its measure (three couriers; two late kitchens; one with cold food),
#   and the three restaurants the two restaurant rankings name together are
#   scored like a top three ("plain means").
# - The same populations with nothing planted (plant=False: the same orders
#   and incidents, left where they fell). The worst group's value there is
#   what chance alone produces. A planted group counts as found with no
#   false alarm when its value is beyond the worst such value in all 20
#   unplanted populations of its size.
#
# Saves notebooks/results/t804_drivers.csv (T804's rows first, unchanged,
# then T811's) and t804_alarms.csv. About sixteen minutes on one CPU core;
# the replays run side by side.

# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fitcore.features import INPUTS
from fitcore.model import SCORES, ModelParams
from fitcore.schemas import load_event
from fitcore.scoring import Customer, apply_event, order_id_for, score
from fitcore.sim_events import PLANTED, simulate_events
from joblib import Parallel, delayed

SEEDS = tuple(range(2026, 2046))
SIZES = (50, 100, 200, 400)
DAYS = 90.0
MIN_ORDERS = 5  # GET /drivers' default
TOP = 3
SPLIT_TOLERANCE = 1e-12  # how far a score's contributions may miss its change
INCIDENT_INPUT = {"late": "x1", "cold_food": "x6", "rude_courier": "x5"}
# T811: the rankings with no model. Each one's column of the orders' table,
# and whether a higher mean is worse.
PLAIN = {
    "mean minutes late": ("minutes_late", True),
    "mean food rating": ("food_rating", False),
    "mean courier rating": ("courier_rating", False),
}
INCIDENT_PLAIN = {
    "late": "mean minutes late",
    "cold_food": "mean food rating",
    "rude_courier": "mean courier rating",
}
UNPLANTED = "worst value, nothing planted"
# Replays side by side: half the cores, as T802 does.
JOBS = max(1, (os.cpu_count() or 2) // 2)


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

# Which input each planted group should show, by grouping; and which planted
# groups each plain ranking should find.
EXPECTED: dict[str, dict[str, str]] = {"restaurant": {}, "courier": {}}
PLAIN_EXPECTED: dict[str, dict[str, tuple[str, ...]]] = {
    "restaurant": {},
    "courier": {},
}
for incident, (field, names) in PLANTED.items():
    by = "restaurant" if field == "restaurant_id" else "courier"
    for name in names:
        EXPECTED[by][name] = INCIDENT_INPUT[incident]
    PLAIN_EXPECTED[by][INCIDENT_PLAIN[incident]] = names


def split_error(scored, contributions: dict[str, dict[str, float]]) -> float:
    """The most any of the six scores' contributions miss its change."""
    return max(
        abs(
            sum(contributions[name].values())
            - (getattr(scored, name) - getattr(as_expected, name))
        )
        for name in SCORES
    )


def minutes_late(order) -> float:
    """How long after its promise an order came, in minutes (early is negative)."""
    if order.delivered_at is None:
        return np.nan
    return (order.delivered_at - order.promised_eta).total_seconds() / 60.0


def scored_orders(events: list[dict]) -> tuple[pd.DataFrame, int, float]:
    """Every scored order's latest contributions to fit, with its groups and
    its raw measures; how many scorings there were, and the largest split
    error among them."""
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
                "minutes_late": minutes_late(orders[order_id]),
                "food_rating": orders[order_id].ratings.get("food", np.nan),
                "courier_rating": orders[order_id].ratings.get("courier", np.nan),
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


def plain_rankings(table: pd.DataFrame, by: str) -> pd.DataFrame:
    """Each group's mean of every raw measure its grouping answers for, and
    its rank by it, worst first (ties by id). Groups with at least MIN_ORDERS
    orders; a group with no such measure has no mean and no rank."""
    sizes = table.groupby(by).size()
    out = pd.DataFrame(index=sizes.index[sizes >= MIN_ORDERS])
    for name in PLAIN_EXPECTED[by]:
        column, higher_is_worse = PLAIN[name]
        means = table.groupby(by)[column].mean().reindex(out.index)
        known = means.dropna()
        out[name] = means
        out[f"rank by {name}"] = _rank(
            -known if higher_is_worse else known, known.index.to_series()
        ).reindex(out.index)
    return out


def worst_by(plain: pd.DataFrame, name: str, count: int) -> list[str]:
    """The count worst groups by one plain ranking."""
    return list(plain[f"rank by {name}"].dropna().sort_values().index[:count])


def _rank(values: pd.Series, ids: pd.Series) -> pd.Series:
    order = sorted(values.index, key=lambda g: (values[g], ids[g]))
    return pd.Series({g: i + 1 for i, g in enumerate(order)}, dtype=float)


def replay(size: int, seed: int, plant: bool) -> dict:
    """One population replayed: T804's rows (when planted), T811's rows, how
    many scorings were checked and the largest split error among them."""
    table, scorings, worst = scored_orders(
        simulate_events(size, seed, DAYS, plant=plant)
    )
    if worst > SPLIT_TOLERANCE:
        raise AssertionError(f"a split misses its score by {worst:g}: {size}, {seed}")

    def row(by, method, measure, group, value) -> dict:
        return {
            "size": size,
            "seed": seed,
            "orders": len(table),
            "by": by,
            "method": method,
            "measure": measure,
            "group": group,
            "value": value,
        }

    rows: list[dict] = []
    extra: list[dict] = []
    if plant:
        rows.append(
            row(None, None, "scorings with exact splits", None, float(scorings))
        )
    for by, expected in EXPECTED.items():
        ranked = rankings(table, by)
        plain = plain_rankings(table, by)
        if not plant:
            # What chance alone produces: the worst group by each ranking.
            first = ranked.sort_values("rank_excess").index[0]
            extra.append(
                row(by, "excess", UNPLANTED, first, ranked.loc[first, "worst_excess"])
            )
            for name in PLAIN_EXPECTED[by]:
                for group in worst_by(plain, name, 1):
                    extra.append(
                        row(by, name, UNPLANTED, group, plain.loc[group, name])
                    )
            continue

        rows.append(row(by, None, "groups ranked", None, float(len(ranked))))
        for method in ("excess", "fit", "drag"):
            top = set(ranked.sort_values(f"rank_{method}").index[:TOP])
            rows.append(
                row(
                    by,
                    method,
                    f"precision at {TOP}",
                    None,
                    len(top & set(expected)) / TOP,
                )
            )
        for group, planted_input in expected.items():
            if group not in ranked.index:  # too few orders to rank
                continue
            g = ranked.loc[group]
            for method in ("excess", "fit", "drag"):
                rows.append(row(by, method, "rank", group, float(g[f"rank_{method}"])))
            for method, column in (("excess", "first_driver"), ("drag", "first_drag")):
                rows.append(
                    row(
                        by,
                        method,
                        "right first driver",
                        group,
                        float(g[column] == planted_input),
                    )
                )
            extra.append(row(by, "excess", "value", group, float(g["worst_excess"])))

        # T811: the rankings with no model, each asked for as many groups as
        # are planted for its measure.
        named: set[str] = set()
        for name, planted in PLAIN_EXPECTED[by].items():
            top = set(worst_by(plain, name, len(planted)))
            named |= top
            extra.append(
                row(
                    by,
                    name,
                    f"precision at {len(planted)}",
                    None,
                    len(top & set(planted)) / len(planted),
                )
            )
            for group in planted:
                if group not in plain.index or np.isnan(plain.loc[group, name]):
                    continue  # too few orders, or nothing measured
                extra.append(
                    row(
                        by,
                        name,
                        "rank",
                        group,
                        float(plain.loc[group, f"rank by {name}"]),
                    )
                )
                extra.append(
                    row(by, name, "value", group, float(plain.loc[group, name]))
                )
        if len(PLAIN_EXPECTED[by]) > 1:  # what they name together, as a top three
            extra.append(
                row(
                    by,
                    "plain means",
                    f"precision at {TOP}",
                    None,
                    len(named & set(expected)) / TOP,
                )
            )
    return {
        "plant": plant,
        "rows": rows,
        "extra": extra,
        "scorings": scorings,
        "worst": worst,
    }


# %% Replay every population, planted and not
started = time.perf_counter()
jobs = [
    (size, seed, plant) for plant in (True, False) for size in SIZES for seed in SEEDS
]
print(f"{len(jobs)} replays, {JOBS} at a time")
done = Parallel(n_jobs=JOBS)(delayed(replay)(*job) for job in jobs)
print(f"all done in {(time.perf_counter() - started) / 60:.0f} min")

# %% Save: T804's rows first, in T804's order, then T811's
RESULTS.mkdir(parents=True, exist_ok=True)
rows = [row for one in done for row in one["rows"]]
extra = [row for one in done for row in one["extra"]]
results = pd.DataFrame(rows + extra)
results.to_csv(RESULTS / "t804_drivers.csv", index=False, float_format="%.6f")

# %% Nothing planted: how far chance alone goes, and who lies beyond it
# The threshold is the worst group's value in the most extreme of the 20
# unplanted populations of a size. A planted group beyond it is found with no
# false alarm among those 20; one that was not ranked counts as not found.
alarms = []
unplanted = results[results["measure"] == UNPLANTED]
values = results[results["measure"] == "value"]
for (size, by, method), chance in unplanted.groupby(
    ["size", "by", "method"], sort=False
):
    higher_is_worse = PLAIN[method][1] if method in PLAIN else False
    planted = values[
        (values["size"] == size) & (values["by"] == by) & (values["method"] == method)
    ]["value"]
    if higher_is_worse:
        threshold = chance["value"].max()
        found = int((planted > threshold).sum())
    else:
        threshold = chance["value"].min()
        found = int((planted < threshold).sum())
    groups = len(PLAIN_EXPECTED[by][method]) if method in PLAIN else len(EXPECTED[by])
    alarms.append(
        {
            "size": size,
            "by": by,
            "method": method,
            "populations": len(chance),
            "threshold": threshold,
            "planted_groups": groups * len(SEEDS),
            "found": found / (groups * len(SEEDS)),
        }
    )
alarm_table = pd.DataFrame(alarms)
alarm_table.to_csv(RESULTS / "t804_alarms.csv", index=False, float_format="%.6f")

# %% Summary: mean over the 20 populations
pd.set_option("display.width", 160)
print(results.groupby("size")["orders"].mean().round(0).rename("orders per population"))
shares = results[~results["measure"].isin(["value", UNPLANTED])]
print(
    shares.pivot_table(
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
    alarm_table.pivot_table(
        index=["by", "method"], columns="size", values="found", sort=False
    ).round(3)
)
checked = sum(one["scorings"] for one in done if one["plant"])
largest = max(one["worst"] for one in done)
print(
    f"{checked:,} scorings, six scores each: every split adds up "
    f"(largest miss {largest:.1e})"
)
for by, method in (("courier", "mean courier rating"), ("restaurant", "plain means")):
    chosen = results[
        (results["by"] == by)
        & (results["method"] == method)
        & (results["measure"] == f"precision at {TOP}")
    ]
    print(
        f"T811 {by}s, {method}, precision at {TOP} by size:",
        " ".join(f"{v:.3f}" for v in chosen.groupby("size")["value"].mean()),
    )
print(np.round(time.perf_counter() - started))
