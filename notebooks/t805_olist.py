# %% [markdown]
# # T805: real orders: is delivery judged by its gap, and does lateness weigh more?
#
# CFI assumes customers judge an order by its gap to what they expected,
# and that a shortfall hurts more than an equal surplus helps. The simulator
# builds both in, so only real data can test them. Olist's public dataset
# (about 100,000 Brazilian e-commerce orders, 2016 to 2018, CC BY-NC-SA 4.0)
# gives each order a promised delivery date, an actual one, a price, a
# freight charge and usually a 1-to-5 review.
#
# Get the data once: download https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# and put olist_orders_dataset.csv, olist_order_items_dataset.csv,
# olist_order_reviews_dataset.csv and olist_customers_dataset.csv in
# notebooks/data/olist/ (git ignores that folder).
#
# 1. The review in a cumulative logit (fitcore.olist) on: the delivery time
#    itself; the gap to the promise, as is; the gap through CFI's asymmetric
#    transform; both gap and time; and the gap plus the fee load. Each is
#    judged by its log-likelihood on 20% of customers held out.
# 2. kappa for delivery: profile likelihood on the training customers, with
#    a 95% likelihood-ratio interval. Then kappa for the fee load, with the
#    delivery gap's kappa held at its best value.
# 3. Does a late first order cost the next one? Customers whose first order
#    was bought at least 180 days before the data ends: did they order again
#    within 180 days, late first orders against the rest?
# 4. When was the review written (T810)? Olist sends its survey when the
#    order arrives or when its promised date passes, whichever is first. So
#    a late order is often reviewed before it is delivered, by a customer
#    who cannot yet know how late it will be. The review by gap is split
#    into reviews written before and after delivery, and kappa is profiled
#    again on the reviews written after delivery, and on those within two
#    weeks of the promise, where early and late orders are both common.
#
# Saves notebooks/results/t805_models.csv, t805_kappa_profile.csv,
# t805_review_by_gap.csv, t805_repeat.csv and t805_variants.csv. About five
# minutes.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from fitcore.baselines import auc
from fitcore.olist import (
    asymmetric,
    fit_ordinal,
    kappa_interval,
    load,
    ordinal_nll,
    profile_kappa,
)
from scipy.stats import norm

SEED = 805
TEST_SHARE = 0.2
CLASSES = 5  # review 1..5, as answers 0..4
KAPPAS = np.unique(
    np.round(
        np.concatenate(
            [np.arange(1.0, 10.01, 0.25), np.arange(11.0, 16.0), [18, 20, 25, 30, 40]]
        ),
        2,
    )
)
FEE_KAPPAS = np.round(
    np.concatenate([np.arange(0.05, 1.0, 0.05), np.arange(1.0, 4.01, 0.25)]), 2
)
REPEAT_DAYS = 180
NEAR_DAYS = 14  # "near the promise": within two weeks of it, early or late
LR_95 = 3.841458820694124 / 2.0  # as fitcore.olist: half the chi-squared(1) quantile


def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


DATA = repo_root() / "notebooks" / "data" / "olist"
RESULTS = repo_root() / "notebooks" / "results"

# %% Load, keep reviewed orders, split by customer
orders = load(DATA)
rated = orders[orders["review"].notna()].reset_index(drop=True)
people = rated["customer"].unique()
held = set(
    np.random.default_rng(SEED).permutation(people)[: round(TEST_SHARE * len(people))]
)
test = rated["customer"].isin(held).to_numpy()
y = (rated["review"].to_numpy() - 1).astype(int)
print(
    f"{len(orders):,} delivered orders, {len(rated):,} with a review, {len(people):,} customers"
)
print(
    f"train {np.sum(~test):,} orders, test {np.sum(test):,}; late: {np.mean(rated['gap_days'] < 0):.1%}"
)

# Scales come from the training customers only, as CFI's do.
gap = rated["gap_days"].to_numpy()
days = rated["days"].to_numpy()
fee = rated["fee"].to_numpy()
low, high = np.quantile(fee[~test], [0.01, 0.99])
fee_gap = np.median(fee[~test]) - np.clip(
    fee, low, high
)  # positive: cheaper than usual
rms = lambda v: float(np.sqrt(np.mean(v**2)))
gap_scale, days_scale, fee_scale = (
    rms(gap[~test]),
    rms(days[~test]),
    rms(fee_gap[~test]),
)

# %% kappa: profile likelihood on the training customers
profile = profile_kappa(
    gap[~test], gap_scale, np.empty((np.sum(~test), 0)), y[~test], CLASSES, KAPPAS
)
kappa, kappa_low, kappa_high = kappa_interval(profile)
symmetric_nll = dict(profile)[1.0]
best_nll = min(nll for _, nll in profile)
print(
    f"delivery kappa {kappa} (95% interval {kappa_low} to {kappa_high}); log-likelihood gain over kappa = 1: {symmetric_nll - best_nll:.1f}"
)

delivery = asymmetric(gap, kappa) / gap_scale
fee_profile = profile_kappa(
    fee_gap[~test],
    fee_scale,
    delivery[~test].reshape(-1, 1),
    y[~test],
    CLASSES,
    FEE_KAPPAS,
)
fee_kappa, fee_low, fee_high = kappa_interval(fee_profile)
print(f"fee kappa {fee_kappa} (95% interval {fee_low} to {fee_high})")

# %% Models, judged on the held-out customers
designs = {
    "no inputs": np.empty((len(y), 0)),
    "delivery time": (days / days_scale).reshape(-1, 1),
    "gap, symmetric": (gap / gap_scale).reshape(-1, 1),
    "gap, asymmetric (CFI's g)": delivery.reshape(-1, 1),
    "gap, asymmetric + delivery time": np.column_stack([delivery, days / days_scale]),
    "gap, asymmetric + fee load": np.column_stack(
        [delivery, asymmetric(fee_gap, fee_kappa) / fee_scale]
    ),
}
rows = []
for name, x in designs.items():
    fit = fit_ordinal(x[~test], y[~test], CLASSES)
    held_nll = ordinal_nll(fit.beta, fit.cuts, x[test], y[test]).mean()
    rows.append(
        {
            "model": name,
            "slopes": " ".join(f"{b:.4f}" for b in fit.beta),
            "train_nll": fit.nll,
            "aic": 2 * fit.nll + 2 * (x.shape[1] + CLASSES - 1),
            "heldout_nll": held_nll,
        }
    )
models = pd.DataFrame(rows)
base = models.loc[models["model"] == "no inputs", "heldout_nll"].item()
models["heldout_gain"] = base - models["heldout_nll"]
models["kappa"] = [np.nan, np.nan, 1.0, kappa, kappa, kappa]
models["fee_kappa"] = [np.nan] * 5 + [fee_kappa]
pd.set_option("display.width", 160)
print(models.round(4))

# %% The review by delivery gap, with the fitted curves (for the paper's figure)
edges = np.array([-np.inf, -20, -10, -6, -4, -2, 0, 2, 4, 6, 10, 15, 20, 30, np.inf])
bins = pd.cut(gap, edges, right=False)
curve_fit = {
    "asymmetric": fit_ordinal(
        designs["gap, asymmetric (CFI's g)"][~test], y[~test], CLASSES
    ),
    "symmetric": fit_ordinal(designs["gap, symmetric"][~test], y[~test], CLASSES),
}


def expected_review(fit, x: np.ndarray) -> np.ndarray:
    """The mean review (1 to 5) a fitted cumulative logit gives each row of x."""
    eta = x @ fit.beta
    below = 1.0 / (1.0 + np.exp(-(fit.cuts[None, :] - eta[:, None])))  # P(answer <= k)
    probs = np.diff(
        np.column_stack([np.zeros(len(eta)), below, np.ones(len(eta))]), axis=1
    )
    return probs @ np.arange(1, CLASSES + 1)


by_gap = (
    pd.DataFrame({"bin": bins, "gap": gap, "review": rated["review"].to_numpy()})
    .groupby("bin", observed=True)
    .agg(
        gap=("gap", "mean"),
        review=("review", "mean"),
        sd=("review", "std"),
        orders=("review", "size"),
    )
    .reset_index()
)
by_gap["ci95"] = 1.96 * by_gap["sd"] / np.sqrt(by_gap["orders"])
by_gap["fitted_asymmetric"] = expected_review(
    curve_fit["asymmetric"],
    (asymmetric(by_gap["gap"].to_numpy(), kappa) / gap_scale).reshape(-1, 1),
)
by_gap["fitted_symmetric"] = expected_review(
    curve_fit["symmetric"], (by_gap["gap"].to_numpy() / gap_scale).reshape(-1, 1)
)
by_gap["bin"] = by_gap["bin"].astype(str)
print(by_gap.round(3))

# %% When was the review written? (T810)
# Olist sends the survey at delivery or on the promised date, whichever
# comes first, so a review can be older than the delivery it is matched to.
after = (rated["reviewed"] >= rated["delivered"]).to_numpy()
late_orders = gap < 0
print(
    f"reviewed before delivery: {np.mean(~after):.1%} of all orders, "
    f"{np.mean(~after[late_orders]):.1%} of late ones, "
    f"{np.mean(~after[~late_orders]):.2%} of the rest"
)
frame = pd.DataFrame(
    {"bin": bins, "review": rated["review"].to_numpy(), "after": after}
)
written_after = frame[frame["after"]].groupby("bin", observed=True)["review"]
written_before = frame[~frame["after"]].groupby("bin", observed=True)["review"]
timing = pd.DataFrame(
    {
        "before_share": 1.0 - frame.groupby("bin", observed=True)["after"].mean(),
        "orders_after": written_after.size(),
        "review_after": written_after.mean(),
        "sd_after": written_after.std(),
        "review_before": written_before.mean(),
    }
).reset_index()
timing["ci95_after"] = 1.96 * timing["sd_after"] / np.sqrt(timing["orders_after"])
by_gap = by_gap.join(timing.drop(columns=["bin", "sd_after"]))
print(by_gap.drop(columns=["sd", "fitted_symmetric"]).round(3))


def interval(profile) -> tuple[float, float]:
    """Where the profile's rise from its minimum crosses LR_95 on each side,
    by linear interpolation between grid points (as paper/figures.py); the
    grid's edge where it never crosses."""
    x = np.array([k for k, _ in profile])
    rise = np.array([v for _, v in profile])
    rise = rise - rise.min()
    best = int(rise.argmin())
    low, high = float(x[0]), float(x[-1])
    for i in range(best, 0, -1):
        if rise[i - 1] > LR_95:
            low = x[i - 1] + (x[i] - x[i - 1]) * (rise[i - 1] - LR_95) / (
                rise[i - 1] - rise[i]
            )
            break
    for i in range(best, len(x) - 1):
        if rise[i + 1] > LR_95:
            high = x[i] + (x[i + 1] - x[i]) * (LR_95 - rise[i]) / (
                rise[i + 1] - rise[i]
            )
            break
    return float(low), float(high)


# kappa again, on the reviews that could know the delivery.
subsets = {
    "all reviews": np.ones(len(rated), dtype=bool),
    "written after delivery": after,
    f"written after delivery, within {NEAR_DAYS} days of the promise": after
    & (np.abs(gap) <= NEAR_DAYS),
}
variant_rows = []
for name, keep in subsets.items():
    fit_rows, held_rows = keep & ~test, keep & test
    scale = rms(gap[fit_rows])
    subset_profile = profile_kappa(
        gap[fit_rows],
        scale,
        np.empty((int(fit_rows.sum()), 0)),
        y[fit_rows],
        CLASSES,
        KAPPAS,
    )
    best, _, _ = kappa_interval(subset_profile)
    low, high = interval(subset_profile)
    columns = {
        "no inputs": np.empty((len(y), 0)),
        "delivery time": (days / rms(days[fit_rows])).reshape(-1, 1),
        "gap, symmetric": (gap / scale).reshape(-1, 1),
        "gap, asymmetric": (asymmetric(gap, best) / scale).reshape(-1, 1),
    }
    held = {}
    for model, x in columns.items():
        fit = fit_ordinal(x[fit_rows], y[fit_rows], CLASSES)
        held[model] = ordinal_nll(fit.beta, fit.cuts, x[held_rows], y[held_rows]).mean()
    variant_rows.append(
        {
            "reviews": name,
            "orders": int(keep.sum()),
            "late_share": float(np.mean(gap[keep] < 0)),
            "kappa": best,
            "kappa_low": low,
            "kappa_high": high,
            "gain_symmetric": held["no inputs"] - held["gap, symmetric"],
            "gain_asymmetric": held["no inputs"] - held["gap, asymmetric"],
            "gain_delivery_time": held["no inputs"] - held["delivery time"],
        }
    )
variants = pd.DataFrame(variant_rows)
print(variants.round(4).to_string())

# %% Does a late first order cost the next one?
ordered = orders.sort_values(["customer", "purchased"]).copy()
ordered["rank"] = ordered.groupby("customer").cumcount()
first = ordered[ordered["rank"] == 0].set_index("customer")
second = ordered[ordered["rank"] == 1].set_index("customer")["purchased"]
cutoff = orders["purchased"].max() - pd.Timedelta(days=REPEAT_DAYS)
eligible = first[first["purchased"] <= cutoff].copy()
wait = (
    eligible.index.map(second).to_series(index=eligible.index) - eligible["purchased"]
)
eligible["again"] = (wait <= pd.Timedelta(days=REPEAT_DAYS)).astype(float)


def wilson(k: float, n: int) -> tuple[float, float]:
    z = norm.ppf(0.975)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


late = eligible["gap_days"] < 0
repeat_rows = []
for name, group in (
    ("late first order", eligible[late]),
    ("on time or early", eligible[~late]),
):
    k, n = group["again"].sum(), len(group)
    lo, hi = wilson(k, n)
    repeat_rows.append(
        {
            "group": name,
            "customers": n,
            "again": int(k),
            "rate": k / n,
            "low": lo,
            "high": hi,
        }
    )
p1, n1 = repeat_rows[0]["rate"], repeat_rows[0]["customers"]
p2, n2 = repeat_rows[1]["rate"], repeat_rows[1]["customers"]
pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
z = (p2 - p1) / np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
repeat = pd.DataFrame(repeat_rows)
repeat["two_sided_p"] = 2 * norm.sf(abs(z))
reviewed = eligible["review"].notna()
repeat["auc_gap"] = auc(
    asymmetric(eligible["gap_days"].to_numpy(), kappa), eligible["again"].to_numpy()
)
repeat["auc_review"] = auc(
    eligible.loc[reviewed, "review"].to_numpy(),
    eligible.loc[reviewed, "again"].to_numpy(),
)
print(repeat.round(4))

# %% Save
RESULTS.mkdir(parents=True, exist_ok=True)
models.to_csv(RESULTS / "t805_models.csv", index=False, float_format="%.6f")
pd.DataFrame(profile, columns=["kappa", "train_nll"]).assign(kind="delivery").pipe(
    lambda d: pd.concat(
        [
            d,
            pd.DataFrame(fee_profile, columns=["kappa", "train_nll"]).assign(
                kind="fee"
            ),
        ]
    )
).to_csv(RESULTS / "t805_kappa_profile.csv", index=False, float_format="%.6f")
by_gap.to_csv(RESULTS / "t805_review_by_gap.csv", index=False, float_format="%.6f")
repeat.to_csv(RESULTS / "t805_repeat.csv", index=False, float_format="%.6f")
variants.to_csv(RESULTS / "t805_variants.csv", index=False, float_format="%.6f")
print(
    f"delivery kappa {kappa} [{kappa_low}, {kappa_high}], fee kappa {fee_kappa} [{fee_low}, {fee_high}]"
)
