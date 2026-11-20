# %% [markdown]
# # T802: CFI against its baselines, in twenty populations
#
# Twenty simulated populations (seeds 2026 to 2045, 2,000 customers each,
# split 80/20 by customer); the first five are T110's. Every model learns
# from the training customers and is judged on the held-out customers'
# orders, on the two labels every order can have: churn within 30 days and
# a reorder within 14.
#
# - Same inputs as CFI, one order's eight gaps: logistic regression on the
#   raw gaps, and on each gap's surplus and shortfall apart (so it can weigh
#   them differently, as CFI's kappa does); gradient boosting with default
#   settings, and with CFI's sign constraints; and an unconstrained network
#   (two hidden layers of 16, its L2 penalty picked on held-out training
#   customers), which may use any connection CFI's theory forbids.
# - History alone: T113's order rate, and BG/NBD (Fader, Hardie and Lee
#   2005) fitted on the training customers' purchase histories.
# - Gaps and history together: gradient boosting on both, and CFI's Fit plus
#   the same history in a logistic regression.
# - A ceiling: the simulator's true Fit plus history, which no model sees.
#
# Then what events can measure (T209): only the inputs v1 events measure
# (x1..x3 always, x5 and x6 on rated orders, x8 on orders with an incident,
# x4 and x7 never), the rest counting as expected. Three CFIs meet them: the
# model above as it is deployed (15 paths, trained with every input
# measured), one retrained on event-measured gaps without the x4 and x7
# paths (13 paths), and the same comparators retrained on those gaps (T810).
# And what the love matrix shows then: each held-out customer's quadrant
# from their latest order, with every input measured, with event inputs
# scored as expected (what the API does), and with each input's last measured
# gap carried forward; each against the simulator's true quadrant (D5).
#
# T810 also adds, for every score, the AUC on the orders whose label window
# closed inside the data (a label can be known early only one way: a next
# order came, so open windows hold no churn and nothing but reorders), a
# calibration error with equal-mass bins, each population's facts (base
# rates, the NPS head, what an unrated order scores, how often a quadrant
# changes), and CFI's paired difference from every comparator.
#
# Saves notebooks/results/t802_baselines.csv, t802_paired.csv,
# t802_facts.csv, t802_display.csv and t802_calibration.csv. One population
# takes about five minutes on one CPU core; populations run side by side.

# %%
import dataclasses
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from fitcore.train import FitModel, export_params, fit_model, intercept_only_losses
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = tuple(range(2026, 2046))
FIRST_FIVE = SEEDS[:5]  # T110's populations; T802's before T810
CUSTOMERS = 2000
TEST_SHARE = 0.2
QUADRANTS = ("lovemark", "brand", "fad", "product")
RESPECT, LOVE, FIT = 0, 1, 4  # columns of the latents and of the truth
NEVER_MEASURED = ("x4", "x7")  # v1 events carry no measure of either (T209)
NETWORK_PENALTIES = (0.1, 1.0, 10.0)
WINDOW_DAYS = {"churn 30d": 30, "reorder 14d": 14}
# A better gap can only raise the chance of a reorder and lower that of churn.
DIRECTION = {"churn 30d": -1, "reorder 14d": 1}
CFI = "CFI (15 paths)"
# Populations side by side: half the cores, since each also runs threads.
JOBS = min(len(SEEDS), max(1, (os.cpu_count() or 2) // 2))


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


def surplus_and_shortfall(gaps: np.ndarray) -> np.ndarray:
    """Each gap as two columns: what it is above 0, and what it is below."""
    return np.column_stack([np.maximum(gaps, 0.0), np.minimum(gaps, 0.0)])


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


def classifier(kind: str, seed: int, outcome: str):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    if kind == "boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    if kind == "monotone boosting":  # the eight gaps only, each with CFI's sign
        return HistGradientBoostingClassifier(
            random_state=seed, monotonic_cst=[DIRECTION[outcome]] * len(INPUTS)
        )
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


def fitted_chance(kind, seed, outcome, x_train, y_train, x_test) -> np.ndarray:
    """A scikit-learn model fitted on the known training labels; its chance of 1 on test."""
    known = ~np.isnan(y_train)
    model = classifier(kind, seed, outcome)
    model.fit(x_train[known], y_train[known].astype(int))
    return model.predict_proba(x_test)[:, 1]


def comparators(seed, outcome, x, y, customer, in_test) -> dict[str, np.ndarray]:
    """Every same-inputs comparator's chance on the held-out orders, trained
    on the eight columns of x (the gaps, however they were measured)."""
    y_train = y[~in_test]
    both = surplus_and_shortfall(x)

    def fitted(kind, columns):
        return fitted_chance(
            kind, seed, outcome, columns[~in_test], y_train, columns[in_test]
        )

    return {
        "logistic regression": fitted("logistic", x),
        "logistic regression, surplus and shortfall": fitted("logistic", both),
        "gradient boosting": fitted("boosting", x),
        "gradient boosting, monotone": fitted("monotone boosting", x),
        "unconstrained network": tuned_network_chance(
            seed, x[~in_test], y_train, customer[~in_test], x[in_test]
        ),
    }


def scores(model: str, inputs: str, outcome: str, p, y, closed, seed: int) -> dict:
    """One model's scores on the held-out orders. closed: the orders whose
    label window ended inside the data, where a label is never known early."""
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
        "ece_mass": ece(p, y, equal_mass=True),
        "auc_closed": auc(p, np.where(closed, y, np.nan)),
    }


def quadrants(params: ModelParams, gaps: np.ndarray) -> np.ndarray:
    latents = forward(params, gaps)
    return np.array([quadrant(params, r, l) for r, l in latents[:, :2]])


def true_quadrants(truth: np.ndarray) -> np.ndarray:
    respect, love = truth[:, RESPECT], truth[:, LOVE]
    return np.where(
        respect >= 0.5,
        np.where(love >= 0.5, "lovemark", "brand"),
        np.where(love >= 0.5, "fad", "product"),
    )


def changes(quadrant_of: np.ndarray, customer: np.ndarray) -> float:
    """How often a customer's quadrant differs from their previous order's
    (rows are customer by customer, in time order)."""
    same_customer = customer[1:] == customer[:-1]
    return float(
        np.mean(quadrant_of[1:][same_customer] != quadrant_of[:-1][same_customer])
    )


# %% One population
def population(seed: int) -> dict:
    """Everything T802 measures in one population, as rows for each file."""
    torch.set_num_threads(1)  # populations run side by side instead
    started = time.perf_counter()
    rows: list[dict] = []
    facts: dict[str, float] = {}
    end = START + timedelta(days=DAYS)

    orders = simulate_orders(simulate_profiles(CUSTOMERS, seed), seed)
    data = from_simulation(orders, simulate_behaviour(orders, seed))
    train, test = split_by_customer(data, TEST_SHARE, seed=seed)
    in_test = held_out(data, seed)
    assert in_test.sum() == len(test)
    customers = np.array(orders.customer_ids)
    h = history(orders.customer_ids, orders.placed_at)
    hist = history_columns(h)
    gaps = data.gaps
    truth = np.asarray(data.truth)
    labels = {"churn 30d": data.churn, "reorder 14d": data.reorder}
    placed = np.array([t.timestamp() for t in orders.placed_at])
    closed = {
        outcome: placed + days * 86_400.0 <= end.timestamp()
        for outcome, days in WINDOW_DAYS.items()
    }
    facts["training customers"] = len(train.customer_ids)
    facts["training orders"] = len(train)
    facts["held-out customers"] = len(test.customer_ids)
    facts["held-out orders"] = len(test)

    # CFI, every input measured.
    params = train_cfi(train)
    latents = forward(params, gaps)
    fit_all = latents[:, FIT]
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
        pd.Series(placed / 86_400.0)[~in_test].groupby(customers[~in_test]).min()
    )
    age = end.timestamp() / 86_400.0 - first_day.loc[last.index].to_numpy()
    bgnbd = fit_bgnbd(last["x"].to_numpy() - 1.0, last["t_x"].to_numpy(), age)
    x_now = h.number[in_test] - 1.0
    t_now = h.days_since_first[in_test]
    bg = {
        "churn 30d": p_no_order(bgnbd, x_now, t_now, t_now, 30.0),
        "reorder 14d": 1.0 - p_no_order(bgnbd, x_now, t_now, t_now, 14.0),
    }

    # What events measure: unmeasured inputs count as expected (0).
    mask = event_mask(orders.incidents, RATING_RATE, seed)
    event_gaps = np.where(mask, gaps, 0.0)
    event_train = dataclasses.replace(train, gaps=event_gaps[~in_test])
    event_params = train_cfi(event_train, events_only=True)
    retrained = outcomes(
        event_params, forward(event_params, event_gaps[in_test])[:, FIT]
    )
    deployed = outcomes(params, forward(params, event_gaps[in_test])[:, FIT])
    on_events = {
        "CFI as deployed (15 paths)": {
            "churn 30d": deployed.churn,
            "reorder 14d": deployed.reorder,
        },
        "CFI retrained (13 paths)": {
            "churn 30d": retrained.churn,
            "reorder 14d": retrained.reorder,
        },
    }

    true_fit = truth[:, FIT]
    calibration = []
    for outcome, y in labels.items():
        y_train, y_test = y[~in_test], y[in_test]
        window_closed = closed[outcome][in_test]

        def logistic(columns, outcome=outcome, y_train=y_train):
            return fitted_chance(
                "logistic", seed, outcome, columns[~in_test], y_train, columns[in_test]
            )

        chance = {(CFI, "gaps"): cfi[outcome]}
        for model, p in comparators(
            seed, outcome, gaps, y, data.customer, in_test
        ).items():
            chance[(model, "gaps")] = p
        chance[("order rate", "history")] = logistic(hist[:, :1])
        chance[("BG/NBD", "history")] = bg[outcome]
        chance[("gradient boosting", "gaps + history")] = fitted_chance(
            "boosting",
            seed,
            outcome,
            np.column_stack([gaps, hist])[~in_test],
            y_train,
            np.column_stack([gaps, hist])[in_test],
        )
        chance[("CFI Fit + history", "gaps + history")] = logistic(
            np.column_stack([fit_all, hist])
        )
        chance[("true Fit + history (ceiling)", "truth + history")] = logistic(
            np.column_stack([true_fit, hist])
        )
        for model, by_outcome in on_events.items():
            chance[(model, "event-measured gaps")] = by_outcome[outcome]
        for model, p in comparators(
            seed, outcome, event_gaps, y, data.customer, in_test
        ).items():
            chance[(model, "event-measured gaps")] = p
        for (model, inputs), p in chance.items():
            rows.append(scores(model, inputs, outcome, p, y_test, window_closed, seed))

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
        # Base rates, and what the open windows hold.
        name = outcome.split()[0]
        facts[f"{name} rate"] = float(np.mean(y_test[known]))
        facts[f"{name} rate, closed windows"] = float(
            np.mean(y_test[known & window_closed])
        )
        facts[f"{name} rate, open windows"] = float(
            np.mean(y_test[known & ~window_closed])
        )
        facts[f"{name} labels from open windows"] = float(
            np.mean(~window_closed[known])
        )
        # The riskiest 1% of orders: what CFI says against what happens.
        risk = cfi[outcome] if outcome == "churn 30d" else 1.0 - cfi[outcome]
        event = y_test if outcome == "churn 30d" else 1.0 - y_test
        top = known & (risk >= np.quantile(risk[known], 0.99))
        facts[f"{name}: riskiest 1%, predicted"] = float(np.mean(risk[top]))
        facts[f"{name}: riskiest 1%, observed"] = float(np.mean(event[top]))

    # The most churn the head can ever say: Fit cannot fall below 0.
    facts["churn head ceiling"] = float(outcomes(params, np.zeros(1)).churn[0])

    # The NPS head, on held-out answers (T810): its log loss against the
    # answer shares alone, and how well Fit ranks the answers.
    nps = data.nps[in_test]
    answered = ~np.isnan(nps)
    chance_of_answer = outcomes(params, fit_all[in_test][answered]).nps[
        np.arange(int(answered.sum())), nps[answered].astype(int)
    ]
    facts["nps log loss"] = float(-np.mean(np.log(chance_of_answer)))
    facts["nps log loss, no inputs"] = intercept_only_losses(train, test)["nps"]
    facts["nps rank correlation with Fit"] = float(
        spearmanr(fit_all[in_test][answered], nps[answered])[0]
    )
    facts["nps rank correlation with the true Fit"] = float(
        spearmanr(true_fit[in_test][answered], nps[answered])[0]
    )

    # What an order scores when it is exactly as expected, and when it has
    # no rating (x5 and x6 unmeasured, but x1 still measured).
    as_expected = forward(params, np.zeros((1, len(INPUTS))))[0]
    facts["respect as expected"] = float(as_expected[RESPECT])
    facts["love as expected"] = float(as_expected[LOVE])
    unrated = in_test & ~mask[:, INPUTS.index("x5")] & ~mask[:, INPUTS.index("x6")]
    respect_unrated = forward(params, event_gaps[unrated])[:, RESPECT]
    for share in (0.05, 0.5, 0.95):
        facts[f"respect of unrated orders, quantile {share}"] = float(
            np.quantile(respect_unrated, share)
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
    truth_q = true_quadrants(truth[latest])
    carried = carry_forward(gaps, mask, orders.customer_ids, h.number)
    rules = {
        "every input measured": gaps,
        "events, unmeasured as expected": event_gaps,
        "events, last measure carried forward": carried,
    }
    churn = data.churn
    display = []
    who = data.customer[test_rows]
    for rule, rule_gaps in rules.items():
        q_latest = quadrants(params, rule_gaps[latest])
        q_orders = quadrants(params, rule_gaps[test_rows])
        row = {
            "seed": seed,
            "rule": rule,
            "agreement": float(np.mean(q_latest == truth_q)),
            "changes": changes(q_orders, who),
        }
        for q in QUADRANTS:
            row[q] = float(np.mean(q_latest == q))
            known = (q_orders == q) & ~np.isnan(churn[test_rows])
            row[f"churn_{q}"] = (
                float(np.mean(churn[test_rows][known])) if known.any() else np.nan
            )
        display.append(row)
    # The simulator's own states have no memory either: how often the true
    # quadrant changes from one order to the next.
    facts["true quadrant changes between orders"] = changes(
        true_quadrants(truth[test_rows]), who
    )

    print(f"seed {seed} done in {time.perf_counter() - started:.0f} s", flush=True)
    return {
        "rows": rows,
        "display": display,
        "calibration": pd.concat(calibration),
        "facts": [
            {"seed": seed, "measure": measure, "value": value}
            for measure, value in facts.items()
        ],
    }


# %% Run every population
started = time.perf_counter()
print(f"{len(SEEDS)} populations, {JOBS} at a time")
done = Parallel(n_jobs=JOBS)(delayed(population)(seed) for seed in SEEDS)
print(f"all done in {(time.perf_counter() - started) / 60:.0f} min")

# %% Save
RESULTS.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame([row for one in done for row in one["rows"]])
results.to_csv(RESULTS / "t802_baselines.csv", index=False, float_format="%.6f")
shown = pd.DataFrame([row for one in done for row in one["display"]])
shown.to_csv(RESULTS / "t802_display.csv", index=False, float_format="%.6f")
facts = pd.DataFrame([row for one in done for row in one["facts"]])
facts.to_csv(RESULTS / "t802_facts.csv", index=False, float_format="%.6f")
pooled = pd.concat([one["calibration"] for one in done])
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

# %% CFI against each comparator, population by population
# A paired difference: both models are judged on the same held-out orders of
# the same population, so what the populations share cancels. Its standard
# error is the differences' standard deviation over the square root of the
# number of populations.
REFERENCES = {
    "gaps": (CFI,),
    "gaps + history": ("CFI Fit + history",),
    "event-measured gaps": ("CFI as deployed (15 paths)", "CFI retrained (13 paths)"),
}
paired = []
for (outcome, inputs), group in results.groupby(["outcome", "inputs"], sort=False):
    if inputs not in REFERENCES:
        continue
    if inputs == "gaps + history":  # the ceiling counts as a comparator here
        ceiling = (results["outcome"] == outcome) & (
            results["inputs"] == "truth + history"
        )
        group = pd.concat([group, results[ceiling]])
    measures = ["auc", "log_loss", "auc_closed"]
    wide = group.pivot(index="seed", columns="model", values=measures)
    for reference in REFERENCES[inputs]:
        for other in group["model"].unique():
            if other in REFERENCES[inputs]:
                continue
            for measure in measures:
                d = wide[(measure, reference)] - wide[(measure, other)]
                better = d < 0 if measure == "log_loss" else d > 0
                paired.append(
                    {
                        "outcome": outcome,
                        "inputs": inputs,
                        "model": reference,
                        "against": other,
                        "measure": measure,
                        "difference": d.mean(),
                        "standard_error": d.std(ddof=1) / np.sqrt(len(d)),
                        "better_in": int(better.sum()),
                        "populations": len(d),
                    }
                )
pairs = pd.DataFrame(paired)
pairs.to_csv(RESULTS / "t802_paired.csv", index=False, float_format="%.6f")

# %% Summary: mean (sd) over the populations
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 50)
summary = results.groupby(["outcome", "inputs", "model"], sort=False)[
    ["auc", "log_loss", "ece", "ece_mass", "auc_closed"]
].agg(["mean", "std"])
print(summary.round(3))
print(pairs[pairs["measure"] == "auc"].drop(columns="measure").round(4).to_string())
print(facts.groupby("measure", sort=False)["value"].agg(["mean", "std"]).round(4))
print(
    shown.groupby("rule", sort=False)
    .mean(numeric_only=True)
    .drop(columns="seed")
    .round(3)
)

# %% The first five populations again, as T802 measured them before T810
first = results[results["seed"].isin(FIRST_FIVE) & (results["outcome"] == "churn 30d")]
check = first.groupby(["inputs", "model"])["auc"].mean()
print(
    "first five populations, churn AUC (before T810: 0.798, 0.796, 0.780, 0.812):",
    " ".join(
        f"{check[key]:.3f}"
        for key in (
            ("gaps", CFI),
            ("gaps", "logistic regression"),
            ("gaps", "gradient boosting"),
            ("gaps + history", "CFI Fit + history"),
        )
    ),
)
