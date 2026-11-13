# %% [markdown]
# # T103: outcomes that respond to fit
#
# Simulates 2,000 customers for half a year with fit driving what they do
# next: how soon they reorder, whether they stop for good, and how they
# answer an NPS survey. The reorder and churn labels are read from each
# customer's order history, the way BigQuery will compute them later.
# Stores everything in `data/sim/`. In VS Code, click "Run Cell" above each
# `# %%` (Ctrl+Enter on the last cell); or run the whole file with
# `python notebooks/t103_outcomes.py`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fitcore.features import INPUTS
from fitcore.sim_orders import simulate_orders, simulate_profiles

N = 2_000
SEED = 2026
BANDS = ["lowest", "low", "middle", "high", "highest"]


def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


def known_share(labels: pd.Series) -> float:
    """The share of True among the labels that are known (not None)."""
    return labels.dropna().astype(bool).mean()


# %%
profiles = simulate_profiles(N, seed=SEED)
orders = pd.DataFrame(simulate_orders(profiles, seed=SEED).columns())
customers = pd.DataFrame(
    {
        "customer_id": [p.customer_id for p in profiles],
        "promise": [p.promise for p in profiles],
        **{f"usual_{key}": [p.usual[key] for p in profiles] for key in INPUTS},
        "incident_rate": [p.incident_rate for p in profiles],
        "days_between_orders": [p.days_between_orders for p in profiles],
    }
)
stopped = orders.groupby("customer_id")["stopped"].any()
print(f"{len(orders):,} orders from {orders['customer_id'].nunique():,} customers")
print(f"customers who stopped within the half year: {stopped.mean():.0%}")
print(f"true fit: mean {orders['fit'].mean():.2f}, sd {orders['fit'].std():.2f}")
for label in ("reorder_14d", "churn_30d"):
    known, true = orders[label].notna().mean(), known_share(orders[label])
    print(f"{label}: known on {known:.1%} of orders, true on {true:.1%}")

# %% [markdown]
# ## Outcomes by fit
#
# T103's "done when": outcome rates respond to fit the way the design says.
# Orders are split into five equal bands by their true fit.

# %%
orders["band"] = pd.qcut(orders["fit"], 5, labels=BANDS)
by_band = orders.groupby("band", observed=True)
table = pd.DataFrame(
    {
        "fit": by_band["fit"].mean(),
        "reorder_14d": by_band["reorder_14d"].agg(known_share),
        "churn_30d": by_band["churn_30d"].agg(known_share),
        "mean_nps": by_band["nps"].agg(lambda s: s.dropna().astype(float).mean()),
        "orders": by_band.size(),
    }
)
table.round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, column, title in zip(
    axes,
    ["reorder_14d", "churn_30d", "mean_nps"],
    ["Reorder within 14 days", "Churn within 30 days", "Mean NPS answer (0-10)"],
    strict=True,
):
    ax.plot(table["fit"], table[column], "o-")
    ax.set_title(title)
    ax.set_xlabel("true fit (band average)")
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, None)
axes[2].set_ylim(0, 10)
fig.suptitle(f"Outcomes by true fit: {N:,} simulated customers, seed {SEED}")
plt.show()

# %% [markdown]
# ## The NPS mix
#
# About one order in five gets a survey. Detractors answer 0-6, passives
# 7-8, promoters 9-10; NPS = % promoters - % detractors.

# %%
answers = orders["nps"].dropna().astype(int)
mix = pd.Series(
    {
        "surveyed": len(answers) / len(orders),
        "detractors": (answers <= 6).mean(),
        "passives": answers.between(7, 8).mean(),
        "promoters": (answers >= 9).mean(),
    }
)
print(f"NPS: {100 * (mix['promoters'] - mix['detractors']):+.0f}")
mix.round(3)

# %% [markdown]
# ## Store them

# %%
folder = repo_root() / "data" / "sim"
folder.mkdir(parents=True, exist_ok=True)
orders.drop(columns="band").to_csv(folder / "t103_orders.csv", index=False)
customers.to_csv(folder / "t103_customers.csv", index=False)
print(f"Saved {len(orders):,} orders and {len(customers):,} customers to {folder}")

# %%
# Read the orders back: the same fit, answers and labels, empty ones included.
again = pd.read_csv(folder / "t103_orders.csv")
assert (again["fit"] == orders["fit"]).all()
for column in ("nps", "reorder_14d", "churn_30d"):
    assert (again[column].isna() == orders[column].isna()).all()
    known = orders[column].notna()
    assert (again.loc[known, column] == orders.loc[known, column]).all()
print("Stored fit, NPS answers and labels match the simulation exactly.")
