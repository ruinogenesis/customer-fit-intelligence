# %% [markdown]
# # T309: the golden replay
#
# One fixed event log, scored the same everywhere. The log is 25 simulated
# customers' first 60 days (fitcore.sim_events, seed 309), frozen in
# notebooks/results/t309_events.jsonl the first time this runs. This
# notebook replays it through fitcore.scoring with T112's model and saves
# what each event scored to t309_golden.json. The tests replay the same log
# through fitcore and through the API, with each store, and compare
# (fitcore/tests/test_golden.py, api/tests/test_golden_api.py).
#
# Re-run it only when a change is meant to move scores, and commit the new
# golden file with that change: its diff shows every score that moved.
# Delete the log first to freeze a new one.
#
# Done when (T309): a fixed event log gives identical scores in the notebook
# and the API, checked on every merge.

# %%
from collections import Counter
from pathlib import Path

from fitcore.golden import read_events, replay, write_events, write_golden
from fitcore.model import ModelParams
from fitcore.sim_events import simulate_events

SEED = 309
CUSTOMERS = 25
DAYS = 60.0


def repo_root() -> Path:
    """The repository's top folder: the nearest folder upwards with a .git."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise FileNotFoundError("run this from inside the repository")


results = repo_root() / "notebooks" / "results"
log = results / "t309_events.jsonl"
model = results / "t112_model.json"

# %% Freeze the log, the first time only
if not log.exists():
    write_events(log, simulate_events(CUSTOMERS, SEED, days=DAYS))
events = read_events(log)
customers = len({event["customer_id"] for event in events})
print(f"{log.name}: {len(events)} events from {customers} customers")
print(dict(Counter(event["type"] for event in events)))

# %% Replay it through fitcore, and save what each event scored
scored = replay(ModelParams.from_json(model.read_bytes()), events)
write_golden(results / "t309_golden.json", model.read_bytes(), scored)
count = sum(result["scores"] is not None for result in scored.values())
print(f"t309_golden.json: {count} of {len(scored)} events scored an order")
