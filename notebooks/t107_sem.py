# %% [markdown]
# # T107: SEM check of every path
#
# Before training the network, the doc's structure is checked as a
# structural equation model (semopy): every path and every anchor, fitted
# on 5,000 simulated customers, one row each. A path that comes out near
# zero or with the wrong sign would be revisited before training.
#
# Missing survey answers are handled by full-information maximum
# likelihood (semopy's ModelMeans), so every customer counts. Every column
# means "more is better" (checkout time and churn are flipped), so every
# expected sign is +. Only signs and significance are read here: semopy's
# overall fit statistics are not meaningful with this missing-data method.
#
# Saves the path table to `notebooks/results/t107_sem_paths.csv`, which is
# committed so the result is kept with the code. In VS Code, click
# "Run Cell" above each `# %%` (Ctrl+Enter on the last cell); or run the
# whole file with `python notebooks/t107_sem.py`.

# %%
from pathlib import Path

import pandas as pd
import semopy
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_customers import customer_table
from fitcore.sim_orders import simulate_orders, simulate_profiles

N = 5_000
SEED = 2026
LATENTS = ["respect", "love", "trust", "convenience", "fit"]

# The doc's structure. "=~" is measured by (the first indicator sets the
# latent's scale); "~" is a path.
MODEL = """
respect =~ average_rating + item_respect
love =~ favorites_saved + promo_free_share + item_love
trust =~ new_restaurant_share + saved_payment + item_trust
convenience =~ quick_checkout + item_convenience
fit =~ reorder + stayed + nps
respect ~ x1 + x4 + x5 + x6
love ~ x6 + x7 + x8
convenience ~ respect + x1 + x2 + x3
trust ~ respect + love
fit ~ trust + convenience
"""


def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


# %%
orders = simulate_orders(simulate_profiles(N, seed=SEED), seed=SEED)
behaviour = simulate_behaviour(orders, seed=SEED)
customers = pd.DataFrame(customer_table(orders, behaviour)).set_index("customer_id")
observed = [c for c in customers.columns if c != "orders" and not c.startswith("true_")]
print(f"{len(orders):,} orders from {len(customers):,} customers")
customers[observed].isna().mean().round(2).to_frame("share missing").transpose()

# %% [markdown]
# ## Fit the model
#
# Every column is standardised first (mean 0, standard deviation 1), so the
# estimates are comparable and the optimiser behaves.

# %%
z = (customers[observed] - customers[observed].mean()) / customers[observed].std()
model = semopy.ModelMeans(MODEL)
result = model.fit(z)
print(f"fitted: {result.success}, after {result.n_it} iterations")

# %% [markdown]
# ## The path table
#
# T107's "done when": a path table with signs and significance. A loading
# fixed at 1 sets its latent's scale, so it has no test of its own.

# %%
estimates = model.inspect()
rows = estimates[(estimates["op"] == "~") & (estimates["rval"] != "1")].copy()
paths = pd.DataFrame(
    {
        "kind": ["path" if lval in LATENTS else "loading" for lval in rows["lval"]],
        "to": rows["lval"],
        "from": rows["rval"],
        "expected": "+",
        "estimate": rows["Estimate"].astype(float).round(3),
        "p_value": pd.to_numeric(rows["p-value"], errors="coerce"),
    }
).reset_index(drop=True)
paths["fixed"] = paths["p_value"].isna()
paths["sign_ok"] = paths["estimate"] > 0
paths["significant"] = paths["fixed"] | (paths["p_value"] < 0.05)
paths.round(3)

# %%
for kind in ("path", "loading"):
    part = paths[paths["kind"] == kind]
    ok = part["sign_ok"].sum()
    free = part[~part["fixed"]]
    strong = (free["p_value"] < 0.001).sum()
    print(
        f"{kind}s: {ok} of {len(part)} with the expected sign; "
        f"{strong} of {len(free)} free ones significant at p < 0.001"
    )
assert paths["sign_ok"].all() and paths["significant"].all()

# %% [markdown]
# ## Save the table

# %%
out = repo_root() / "notebooks" / "results" / "t107_sem_paths.csv"
out.parent.mkdir(parents=True, exist_ok=True)
paths.to_csv(out, index=False, float_format="%.6g")
print(f"Saved {len(paths)} rows to {out}")
