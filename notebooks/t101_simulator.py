# %% [markdown]
# # T101: simulated customers with known latents
#
# Builds a seeded population with `fitcore.sim`, looks at how the true Love,
# Respect, Trust and Convenience are spread, and stores them in
# `data/sim/t101_customers.csv`. In VS Code, click "Run Cell" above each
# `# %%`; or run the whole file with `python notebooks/t101_simulator.py`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fitcore.sim import simulate_customers

N = 10_000
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
pop = simulate_customers(N, seed=SEED)
customers = pd.DataFrame(pop.columns())
customers.head()

# %% [markdown]
# ## How the true latents are spread

# %%
customers[LATENTS].describe().round(3)

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for ax, name in zip(axes, LATENTS, strict=True):
    ax.hist(customers[name], bins=40, range=(0.0, 1.0))
    ax.set_title(name)
fig.suptitle(f"True latents: {N:,} simulated customers, seed {SEED}")
plt.show()

# %% [markdown]
# ## How the latents relate
#
# Trust is built from Respect and Love, and Convenience from Respect, so those
# pairs should correlate. Respect and Love share only x6, so theirs is weak.

# %%
customers[LATENTS].corr().round(2)

# %% [markdown]
# ## Store them

# %%
out = repo_root() / "data" / "sim" / "t101_customers.csv"
out.parent.mkdir(parents=True, exist_ok=True)
customers.to_csv(out, index=False)
print(f"Saved {len(customers):,} customers to {out}")

# %%
# Read the file back: the same customers, with the same true latents.
again = pd.read_csv(out)
assert again["customer_id"].tolist() == list(pop.customer_ids)
assert (again[LATENTS] == customers[LATENTS]).all().all()
print("Stored true latents match the simulation exactly.")
