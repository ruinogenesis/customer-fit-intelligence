# %% [markdown]
# # T806: what one event costs, in time and in work
#
# Two parts, on the same events: seed 806's 50 simulated customers over
# their first 30 days (743 events), new customers on dev.
#
# - Here, in this process: the API with its memory store and outbox, as the
#   tests run it. For each event: the documents it reads and saves, and the
#   bytes it sends on to BigQuery. With the registry on (as on dev), every
#   event also reads the live model's document; that read is added here,
#   since a model file stands in for the registry in this process.
# - On dev: the same events sent one after another through Firebase Hosting
#   to Cloud Run (1 vCPU, 512 MiB, asia-southeast3), each timed from this
#   computer; then Cloud Run's own latency for each request, read back from
#   Cloud Logging.
#
# Run from the repo root, with dev's write key in the environment and
# gcloud signed in:
#
#     export CFI_WRITE_KEY=$(gcloud secrets versions access latest \
#       --secret cfi-write-key --project cfi-dev-th26) \
#       && python notebooks/t806_system.py; unset CFI_WRITE_KEY
#
# The key is only sent in a header; nothing here prints it. Sending the run
# again changes nothing on dev (every event_id is fixed) and times the
# already-received path instead, so the script refuses unless every event
# is new. Saves notebooks/results/t806_system.csv. About three minutes.

# %%
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from cfi_api.app import create_app
from cfi_api.export import MemoryPublisher
from cfi_api.fake_orders import http_sender
from cfi_api.store import MemoryStore
from fitcore.sim_events import simulate_events

SEED = 806
CUSTOMERS = 50
DAYS = 30.0
URL = "https://cfi-dev-th26.web.app"
PROJECT = "cfi-dev-th26"
SERVICE = "cfi-api"
REGISTRY_READS = 1  # the live model's document, read inside every event (T310)
LOG_WAIT = 180.0  # seconds to wait for Cloud Logging to hold every request


def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


RESULTS = repo_root() / "notebooks" / "results"
MODEL = RESULTS / "t112_model.json"


class CountingStore(MemoryStore):
    """A memory store that counts the documents each event reads."""

    def __init__(self) -> None:
        super().__init__(outbox=True)
        self.reads = 0

    def find_event(self, event_id):
        self.reads += 1
        return super().find_event(event_id)

    def find_customer(self, customer_id):
        self.reads += 1
        return super().find_customer(customer_id)

    def find_order(self, order_id):
        self.reads += 1
        return super().find_order(order_id)


def percentiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def cloud_run_latencies(since: datetime, until: datetime) -> list[float]:
    """Cloud Run's own latency, in ms, of each POST /api/v1/events in the span."""
    span = f'timestamp>="{since.isoformat()}" AND timestamp<="{until.isoformat()}"'
    query = (
        'resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{SERVICE}" '
        'AND httpRequest.requestMethod="POST" '
        'AND httpRequest.requestUrl:"/api/v1/events" '
        f"AND {span}"
    )
    out = subprocess.run(
        [
            "gcloud",
            "logging",
            "read",
            query,
            "--project",
            PROJECT,
            "--format",
            "json(httpRequest.latency)",
            "--limit",
            "10000",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found = json.loads(out or "[]")
    return [
        1000.0 * float(entry["httpRequest"]["latency"].rstrip("s"))
        for entry in found
        if entry.get("httpRequest", {}).get("latency")
    ]


# %% The events
events = simulate_events(CUSTOMERS, SEED, DAYS)
kinds = Counter(event["type"] for event in events)
print(f"{len(events)} events from {CUSTOMERS} customers over {DAYS:g} days")
print(dict(kinds))

# %% Here: what each event reads, saves and sends on
# create_app reads CFI_* settings from the environment; this app is the
# tests' one, so it must not pick up dev's write key.
dev_settings = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("CFI_")}
try:
    app = create_app({"MODEL_PATH": str(MODEL), "MODEL_VERSION": "t112-sim"})
finally:
    os.environ.update(dev_settings)
store = app.extensions["cfi_store"] = CountingStore()
publisher = app.extensions["cfi_publisher"] = MemoryPublisher()
client = app.test_client()
rows: list[dict] = []
local_ms: list[float] = []
for event in events:
    reads, sent = store.reads, len(publisher.sent)
    started = time.perf_counter()
    answer = client.post("/api/v1/events", json=event)
    local_ms.append(1000.0 * (time.perf_counter() - started))
    if answer.status_code != 201:
        sys.exit(
            f"{event['type']} got {answer.status_code} here: {answer.get_data(True)}"
        )
    messages = publisher.sent[sent:]
    rows.append(
        {
            "type": event["type"],
            "reads": store.reads - reads + REGISTRY_READS,
            "documents": len(messages),
            "bytes": sum(len(m.encode()) for m in messages),
        }
    )
work = pd.DataFrame(rows)
print(work.groupby("type")[["reads", "documents", "bytes"]].mean().round(2))
orders = kinds["order_placed"]
per_event = {
    # The outbox entry is one more write in the event's transaction, and one
    # delete once Pub/Sub has it; the event's own document is deleted by TTL.
    "reads": work["reads"].mean(),
    "writes": work["documents"].mean() + 1.0,
    "deletes": 2.0,
    "messages": work["documents"].mean(),
    "bigquery bytes": work["bytes"].mean(),
}
print({key: round(value, 2) for key, value in per_event.items()})
print(f"BigQuery bytes per order: {work['bytes'].sum() / orders:.0f}")
print(f"in this process: {percentiles(local_ms)} ms")

# %% Dev: the same events, one after another, timed from here
key = os.environ.get("CFI_WRITE_KEY")
if not key:
    sys.exit("Set CFI_WRITE_KEY first (see the top of this file).")
send = http_sender(URL, key)
# Wake an instance first, so the first event does not pay for a cold start.
for _ in range(3):
    with urllib.request.urlopen(URL + "/api/v1/health", timeout=30) as response:
        response.read()
since = datetime.now(UTC) - timedelta(seconds=5)
remote_ms: list[float] = []
statuses: Counter[int] = Counter()
for event in events:
    started = time.perf_counter()
    status, body = send(event)
    remote_ms.append(1000.0 * (time.perf_counter() - started))
    statuses[status] += 1
    if status not in (200, 201):
        sys.exit(f"{event['type']} got {status} on dev: {body.strip()[:200]}")
until = datetime.now(UTC) + timedelta(seconds=5)
if statuses[200]:
    sys.exit(
        f"{statuses[200]} events were already on dev, so these times are not "
        "for new events. Change SEED to a seed dev has not seen."
    )
print(f"from here, every event: {percentiles(remote_ms)} ms")

# %% Dev: Cloud Run's own latency, once Cloud Logging has every request
waited = 0.0
server_ms = cloud_run_latencies(since, until)
while len(server_ms) < len(events) and waited < LOG_WAIT:
    time.sleep(15.0)
    waited += 15.0
    server_ms = cloud_run_latencies(since, until)
print(f"Cloud Run logged {len(server_ms)} of {len(events)} requests")
print(f"in Cloud Run: {percentiles(server_ms)} ms")

# %% Save
out = [
    {"part": "events", "measure": "count", "value": len(events)},
    {"part": "events", "measure": "orders", "value": orders},
    *(
        {"part": "per event", "measure": name, "value": value}
        for name, value in per_event.items()
    ),
    {
        "part": "per order",
        "measure": "bigquery bytes",
        "value": work["bytes"].sum() / orders,
    },
    *(
        {"part": where, "measure": name, "value": value}
        for where, values in (
            ("ms in this process", local_ms),
            ("ms from here", remote_ms),
            ("ms in cloud run", server_ms),
        )
        for name, value in percentiles(values).items()
    ),
    {"part": "ms in cloud run", "measure": "logged", "value": len(server_ms)},
]
RESULTS.mkdir(parents=True, exist_ok=True)
pd.DataFrame(out).to_csv(RESULTS / "t806_system.csv", index=False, float_format="%.6g")
print(f"saved {len(out)} rows to notebooks/results/t806_system.csv")
