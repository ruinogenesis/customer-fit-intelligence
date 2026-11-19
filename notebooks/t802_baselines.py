# %% [markdown]
# # T802: CFI against its baselines, in five populations
#
# The five populations of T110 (seeds 2026 to 2030, 2,000 customers each,
# split 80/20 by customer). Every model learns from the training customers
# and is judged on the held-out customers' orders, on the two labels every
# order can have: churn within 30 days and a reorder within 14.
#
# - Same inputs as CFI, one order's eight gaps: logistic regression,
#   gradient boosting, and an unconstrained network (two hidden layers of
#   16, its L2 penalty picked on held-out training customers), which may use
#   any connection CFI's theory forbids.
# - History alone: T113's order rate, and BG/NBD (Fader, Hardie and Lee
#   2005) fitted on the training customers' purchase histories.
# - Gaps and history together: gradient boosting on both, and CFI's Fit plus
#   the same history in a logistic regression.
# - A ceiling: the simulator's true Fit plus history, which no model sees.
#
# Then what events can measure (T209): CFI trained and judged with only the
# inputs v1 events measure (x1..x3 always, x5 and x6 on rated orders, x8 on
# orders with an incident, x4 and x7 never), the rest counting as expected.
# And what the love matrix shows then: each held-out customer's quadrant
# from their latest order, with every input measured, with event inputs
# scored as expected (what the API does), and with each input's last measured
# gap carried forward; each against the simulator's true quadrant (D5).
#
# Saves notebooks/results/t802_baselines.csv, t802_display.csv and
# t802_calibration.csv. About fifteen minutes on two CPU cores.

# %%
import dataclasses
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fitcore.baselines import (
    auc,
    brier,
    carry_forward,
    ece,
    event_mask,
    fit_bgnbd,
    history,
    log_loss,
    p_no_order,
)
from fitcore.dataset import TrainingSet, from_simulation, split_by_customer
from fitcore.features import INPUTS
from fitcore.model import PATHS, ModelParams, forward, masks_without, outcomes, quadrant
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_events import RATING_RATE
from fitcore.sim_orders import DAYS, START, simulate_orders, simulate_profiles
from fitcore.train import FitModel, export_params, fit_model
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (2026, 2027, 2028, 2029, 2030)
CUSTOMERS = 2000
TEST_SHARE = 0.2
QUADRANTS = ("lovemark", "brand", "fad", "product")
FIT = 4  # Fit's column in the latents and in the simulator's truth
NEVER_MEASURED = ("x4", "x7")  # v1 events carry no measure of either (T209)
NETWORK_PENALTIES = (0.1, 1.0, 10.0)


# %% Helpers
def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


RESULTS = repo_root() / "notebooks" / "results"


def held_out(data: TrainingSet, seed: int) -> np.ndarray:
    """Which rows split_by_customer puts in test: the same draw, as a row mask."""
    count = len(data.customer_ids)
    chosen = np.random.default_rng(seed).permutation(count)[: round(TEST_SHARE * count)]
    return np.isin(data.customer, chosen)


def history_columns(h) -> np.ndarray:
    """The history every history model reads: order rate, first order, days since the last."""
    first = np.isnan(h.days_since_previous)
    return np.column_stack(
        [
            h.log_rate,
            first.astype(float),
            np.log1p(np.nan_to_num(h.days_since_previous)),
        ]
    )


def train_cfi(train: TrainingSet, events_only: bool = False) -> ModelParams:
    """CFI fitted on train. events_only: x4 and x7 are never measured, so
    their paths are switched off and their scale (unused) set to 1; every
    other input keeps its root mean square, unmeasured zeros included, as a
    model trained on event data would have."""
    if events_only:
        never = [INPUTS.index(key) for key in NEVER_MEASURED]
        scale = np.sqrt(np.mean(train.gaps**2, axis=0))
        scale[never] = 1.0
        masks = masks_without(p for p in PATHS if p.source in NEVER_MEASURED)
        behaviours = train.behaviours
        model = FitModel(scale, behaviours.mean(axis=0), behaviours.std(axis=0), masks)
    else:
        model = FitModel.for_training_set(train)
    result = fit_model(model, train)
    if not result.converged:
        raise RuntimeError("CFI did not converge")
    return export_params(model)


def classifier(kind: str, seed: int):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(kind)


def network(alpha: float, seed: int):
    """Two hidden layers of 16, fitted by L-BFGS like CFI, with an L2 penalty alpha."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 16),
            solver="lbfgs",
            alpha=alpha,
            max_iter=2000,
            random_state=seed,
        ),
    )


def tuned_network_chance(seed, x_train, y_train, customer_train, x_test) -> np.ndarray:
    """The unconstrained network, its penalty picked on a fifth of the training
    customers (by held-out log loss), then refitted on them all."""
    known = ~np.isnan(y_train)
    x, y, who = x_train[known], y_train[known].astype(int), customer_train[known]
    people = np.unique(who)
    inner = np.isin(
        who, np.random.default_rng(seed).permutation(people)[: len(people) // 5]
    )
    losses = {}
    for alpha in NETWORK_PENALTIES:
        p = network(alpha, seed).fit(x[~inner], y[~inner]).predict_proba(x[inner])[:, 1]
        losses[alpha] = log_loss(p, y[inner])
    best = min(losses, key=losses.__getitem__)
    return network(best, seed).fit(x, y).predict_proba(x_test)[:, 1]


def fitted_chance(kind, seed, x_train, y_train, x_test) -> np.ndarray:
    """A scikit-learn model fitted on the known training labels; its chance of 1 on test."""
    known = ~np.isnan(y_train)
    model = classifier(kind, seed).fit(x_train[known], y_train[known].astype(int))
    return model.predict_proba(x_test)[:, 1]


def scores(model: str, inputs: str, outcome: str, p, y, seed: int) -> dict:
    return {
        "seed": seed,
        "model": model,
        "inputs": inputs,
        "outcome": outcome,
        "n": int((~np.isnan(y)).sum()),
        "auc": auc(p, y),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "ece": ece(p, y),
    }


def quadrants(params: ModelParams, gaps: np.ndarray) -> np.ndarray:
    latents = forward(params, gaps)
    return np.array([quadrant(params, r, l) for r, l in latents[:, :2]])


def true_quadrants(truth: np.ndarray) -> np.ndarray:
    respect, love = truth[:, 0], truth[:, 1]
    return np.where(
        respect >= 0.5,
        np.where(love >= 0.5, "lovemark", "brand"),
        np.where(love >= 0.5, "fad", "product"),
    )


# %% Run every population
rows: list[dict] = []
display: list[dict] = []
calibration: list[pd.DataFrame] = []
end = START + timedelta(days=DAYS)
started = time.perf_counter()

for seed in SEEDS:
    orders = simulate_orders(simulate_profiles(CUSTOMERS, seed), seed)
    data = from_simulation(orders, simulate_behaviour(orders, seed))
    train, test = split_by_customer(data, TEST_SHARE, seed=seed)
    in_test = held_out(data, seed)
    assert in_test.sum() == len(test)
    customers = np.array(orders.customer_ids)
    h = history(orders.customer_ids, orders.placed_at)
    hist = history_columns(h)
    gaps = data.gaps
    labels = {"churn 30d": data.churn, "reorder 14d": data.reorder}

    # CFI, every input measured.
    params = train_cfi(train)
    fit_all = forward(params, gaps)[:, FIT]
    heads = outcomes(params, fit_all[in_test])
    cfi = {"churn 30d": heads.churn, "reorder 14d": heads.reorder}

    # BG/NBD: fitted on each training customer's whole history at the end.
    by_customer = pd.DataFrame(
        {"customer": customers, "number": h.number, "since_first": h.days_since_first}
    )
    last = (
        by_customer[~in_test]
        .groupby("customer")
        .agg(x=("number", "max"), t_x=("since_first", "max"))
    )
    first_day = (
        pd.Series([t.timestamp() / 86_400.0 for t in orders.placed_at])[~in_test]
        .groupby(customers[~in_test])
        .min()
    )
    age = end.timestamp() / 86_400.0 - first_day.loc[last.index].to_numpy()
    bgnbd = fit_bgnbd(last["x"].to_numpy() - 1.0, last["t_x"].to_numpy(), age)
    x_now = h.number[in_test] - 1.0
    t_now = h.days_since_first[in_test]
    bg = {
        "churn 30d": p_no_order(bgnbd, x_now, t_now, t_now, 30.0),
        "reorder 14d": 1.0 - p_no_order(bgnbd, x_now, t_now, t_now, 14.0),
    }

    true_fit = np.asarray(data.truth)[:, FIT]
    for outcome, y in labels.items():
        y_train, y_test = y[~in_test], y[in_test]
        chance = {
            ("CFI (15 paths)", "gaps"): cfi[outcome],
            ("logistic regression", "gaps"): fitted_chance(
                "logistic", seed, gaps[~in_test], y_train, gaps[in_test]
            ),
            ("gradient boosting", "gaps"): fitted_chance(
                "boosting", seed, gaps[~in_test], y_train, gaps[in_test]
            ),
            ("unconstrained network", "gaps"): tuned_network_chance(
                seed, gaps[~in_test], y_train, data.customer[~in_test], gaps[in_test]
            ),
            ("order rate", "history"): fitted_chance(
                "logistic", seed, hist[~in_test, :1], y_train, hist[in_test, :1]
            ),
            ("BG/NBD", "history"): bg[outcome],
            ("gradient boosting", "gaps + history"): fitted_chance(
                "boosting",
                seed,
                np.column_stack([gaps, hist])[~in_test],
                y_train,
                np.column_stack([gaps, hist])[in_test],
            ),
            ("CFI Fit + history", "gaps + history"): fitted_chance(
                "logistic",
                seed,
                np.column_stack([fit_all, hist])[~in_test],
                y_train,
                np.column_stack([fit_all, hist])[in_test],
            ),
            ("true Fit + history (ceiling)", "truth + history"): fitted_chance(
                "logistic",
                seed,
                np.column_stack([true_fit, hist])[~in_test],
                y_train,
                np.column_stack([true_fit, hist])[in_test],
            ),
        }
        for (model, inputs), p in chance.items():
            rows.append(scores(model, inputs, outcome, p, y_test, seed))
        known = ~np.isnan(y_test)
        bins = np.clip((cfi[outcome][known] * 10).astype(int), 0, 9)
        calibration.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "outcome": outcome,
                    "bin": bins,
                    "predicted": cfi[outcome][known],
                    "observed": y_test[known],
                }
            )
        )

    # CFI on what events measure: unmeasured inputs count as expected (0).
    mask = event_mask(orders.incidents, RATING_RATE, seed)
    event_gaps = np.where(mask, gaps, 0.0)
    event_train = dataclasses.replace(train, gaps=event_gaps[~in_test])
    event_params = train_cfi(event_train, events_only=True)
    event_heads = outcomes(
        event_params, forward(event_params, event_gaps[in_test])[:, FIT]
    )
    for outcome, p in (
        ("churn 30d", event_heads.churn),
        ("reorder 14d", event_heads.reorder),
    ):
        rows.append(
            scores(
                "CFI (15 paths)",
                "event-measured gaps",
                outcome,
                p,
                labels[outcome][in_test],
                seed,
            )
        )

    # What the love matrix shows: each held-out customer's latest order.
    test_rows = np.flatnonzero(in_test)
    latest = (
        pd.DataFrame(
            {
                "row": test_rows,
                "customer": customers[test_rows],
                "n": h.number[test_rows],
            }
        )
        .sort_values("n")
        .groupby("customer")["row"]
        .last()
        .to_numpy()
    )
    truth_q = true_quadrants(np.asarray(data.truth)[latest])
    carried = carry_forward(gaps, mask, orders.customer_ids, h.number)
    rules = {
        "every input measured": gaps,
        "events, unmeasured as expected": event_gaps,
        "events, last measure carried forward": carried,
    }
    churn = data.churn
    for rule, rule_gaps in rules.items():
        q_latest = quadrants(params, rule_gaps[latest])
        q_orders = quadrants(params, rule_gaps[test_rows])
        row = {
            "seed": seed,
            "rule": rule,
            "agreement": float(np.mean(q_latest == truth_q)),
        }
        for q in QUADRANTS:
            row[q] = float(np.mean(q_latest == q))
            known = (q_orders == q) & ~np.isnan(churn[test_rows])
            row[f"churn_{q}"] = (
                float(np.mean(churn[test_rows][known])) if known.any() else np.nan
            )
        display.append(row)

    print(f"seed {seed} done, {time.perf_counter() - started:.0f} s so far")

# %% Save
RESULTS.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(rows)
results.to_csv(RESULTS / "t802_baselines.csv", index=False, float_format="%.6f")
shown = pd.DataFrame(display)
shown.to_csv(RESULTS / "t802_display.csv", index=False, float_format="%.6f")
pooled = pd.concat(calibration)
curve = (
    pooled.groupby(["outcome", "bin"])
    .agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        orders=("observed", "size"),
    )
    .reset_index()
)
curve.to_csv(RESULTS / "t802_calibration.csv", index=False, float_format="%.6f")

# %% Summary: mean (sd) over the five populations
summary = results.groupby(["outcome", "inputs", "model"], sort=False)[
    ["auc", "log_loss", "brier", "ece"]
].agg(["mean", "std"])
pd.set_option("display.width", 160)
print(summary.round(3))
print(
    shown.groupby("rule", sort=False)
    .mean(numeric_only=True)
    .drop(columns="seed")
    .round(3)
)
