# %% [markdown]
# # T102: simulated order histories and their gaps
#
# Gives 2,000 simulated customers half a year of orders. Every order is
# compared with what that customer expected just before it (x1 with the
# promise, x2..x8 with `fitcore.expectations`), giving all eight gaps and
# the true latents behind them. Stores the orders and the customers' service
# profiles in `data/sim/`. In VS Code, click "Run Cell" above each `# %%`
# (Ctrl+Enter runs a cell without adding a new one at the end); or run the
# whole file with `python notebooks/t102_orders.py`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitcore.features import INPUTS
from fitcore.sim_orders import simulate_orders, simulate_profiles

N = 2_000
SEED = 2026
LATENTS = ["respect", "love", "convenience", "trust"]


def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


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
per_customer = orders.groupby("customer_id").size()
print(f"{len(orders):,} orders from {orders['customer_id'].nunique():,} customers")
print(f"orders per customer: median {per_customer.median():.0f}")
print(f"orders with an incident: {(orders['incident'] != 'none').mean():.1%}")
orders.head()

# %% [markdown]
# ## Every order has all eight gaps
#
# T102's "done when": no order is missing a gap.

# %%
assert orders[list(INPUTS)].notna().all().all()
orders[list(INPUTS)].describe().round(3)

# %% [markdown]
# ## One customer's history
#
# Food rating (x6) against what this customer expected, and the true Respect
# behind each order. Incidents are marked. Watch the expectation dip after a
# bad order, and the next normal order feel like a small relief.

# %%
busy = per_customer[per_customer >= 25].index
rates = customers.set_index("customer_id")["incident_rate"]
pick = rates[busy].idxmax()
mine = orders[orders["customer_id"] == pick]
hit = mine["incident"] != "none"

fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
top.plot(mine["placed_at"], mine["x6_measured"], "o-", label="food rating")
top.plot(mine["placed_at"], mine["x6_expected"], "--", label="expected")
top.set_ylabel("x6, 1 to 5")
top.legend()
bottom.plot(mine["placed_at"], mine["respect"], "o-", label="true Respect")
bottom.plot(
    mine.loc[hit, "placed_at"], mine.loc[hit, "respect"], "rx", ms=10, label="incident"
)
bottom.set_ylim(0, 1)
bottom.set_ylabel("Respect")
bottom.legend()
fig.suptitle(f"{pick}: incident rate {rates[pick]:.0%}, {len(mine)} orders")
plt.show()

# %% [markdown]
# ## What incidents do to Respect and Love
#
# Average true latents on orders with each kind of incident. Poor support
# hits x8, which feeds Love but not Respect, so it shows only in Love.

# %%
by_incident = orders.groupby("incident")
by_incident[["respect", "love"]].mean().round(3).assign(orders=by_incident.size())

# %% [markdown]
# ## Customers with more incidents have lower Respect
#
# Expectations adapt to each customer's usual service, so steady differences
# fade. Incidents stay surprising, so a customer's incident rate still shows
# in their average Respect: a real but modest link.

# %%
summary = customers.set_index("customer_id").join(
    orders.groupby("customer_id")[LATENTS].mean()
)
r = np.corrcoef(summary["incident_rate"], summary["respect"])[0, 1]
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(summary["incident_rate"], summary["respect"], s=4, alpha=0.4)
ax.set_xlabel("incident rate")
ax.set_ylabel("average true Respect")
ax.set_title(f"{N:,} customers, correlation {r:.2f}")
plt.show()

# %% [markdown]
# ## Store them

# %%
folder = repo_root() / "data" / "sim"
folder.mkdir(parents=True, exist_ok=True)
orders.to_csv(folder / "t102_orders.csv", index=False)
customers.to_csv(folder / "t102_customers.csv", index=False)
print(f"Saved {len(orders):,} orders and {len(customers):,} customers to {folder}")

# %%
# Read the orders back: the same gaps and true latents, exactly.
again = pd.read_csv(folder / "t102_orders.csv")
assert again["customer_id"].tolist() == orders["customer_id"].tolist()
assert (again[list(INPUTS) + LATENTS] == orders[list(INPUTS) + LATENTS]).all().all()
print("Stored gaps and true latents match the simulation exactly.")
