"""The golden replay (T309): one fixed event log, scored the same everywhere.

notebooks/results/t309_events.jsonl is a frozen event log, one raw event a
line: 25 simulated customers' first 60 days (fitcore.sim_events, seed 309).
notebooks/t309_golden.py replays it through fitcore.scoring with T112's
model and saves what each event scored to t309_golden.json. The tests
replay the same log through fitcore (fitcore/tests/test_golden.py) and
through the API with each store (api/tests/test_golden_api.py), and compare.

A result is what POST /events answers about scores: the order the event
scored, that order's scores and fit's top drivers (T316), or None for all
three. Results are compared number by number within TOLERANCE, everything
else exactly. The tolerance
is for another machine adding up in another order: in a trial, that moved
a latent by at most 4.4e-16, while the closest score to a quadrant line was
7.7e-3 away.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fitcore.model import ModelParams
from fitcore.schemas import load_event
from fitcore.scoring import Customer, Order, apply_event, order_id_for

TOLERANCE = 1e-12

# One event's result: {"order_id": str | None, "scores": dict | None,
# "drivers": list | None}.
Result = dict[str, Any]


@dataclass(frozen=True)
class Golden:
    """The expected results, by event_id in log order, and the sha256 of
    the model.json they were scored with."""

    model_sha256: str
    results: dict[str, Result]


def read_events(path: Path) -> list[dict[str, Any]]:
    """The raw events of a JSON Lines file, in order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_events(path: Path, events: Iterable[Mapping[str, Any]]) -> None:
    """Write raw events as JSON Lines: one event a line, keys sorted."""
    lines = [json.dumps(event, sort_keys=True) + "\n" for event in events]
    path.write_text("".join(lines))


def model_sha256(model: bytes) -> str:
    """The sha256 of a model.json's bytes, as hex."""
    return hashlib.sha256(model).hexdigest()


def replay(
    params: ModelParams, events: Iterable[Mapping[str, Any]]
) -> dict[str, Result]:
    """What each raw event scores, replayed in order through fitcore.scoring.

    Keeps customers and orders in dicts, as a notebook would, and answers
    each event's result by event_id, as POST /events would.
    """
    customers: dict[str, Customer] = {}
    orders: dict[str, Order] = {}
    results: dict[str, Result] = {}
    for raw in events:
        event = load_event(raw)
        customer_id = event["customer_id"]
        customer = customers.get(customer_id) or Customer.new(customer_id)
        wanted = order_id_for(event, customer)
        order = None if wanted is None else orders.get(wanted)
        applied = apply_event(params, event, customer, order)
        customers[customer_id] = applied.customer
        if applied.order is not None:
            orders[applied.order.order_id] = applied.order
        scored, explained = applied.scored, applied.explained
        results[str(event["event_id"])] = {
            "order_id": applied.order.order_id if scored and applied.order else None,
            "scores": None if scored is None else scored.to_dict(),
            "drivers": None if explained is None else list(explained.drivers),
        }
    return results


def write_golden(path: Path, model: bytes, results: Mapping[str, Result]) -> None:
    """Save results as a golden file, one event a line so diffs stay readable."""
    lines = [
        f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in results.items()
    ]
    head = f'{{\n "model_sha256": "{model_sha256(model)}",\n "results": {{\n'
    path.write_text(head + ",\n".join(lines) + "\n }\n}\n")


def read_golden(path: Path) -> Golden:
    document = json.loads(path.read_text())
    return Golden(document["model_sha256"], document["results"])


def _same(got: Any, want: Any, tolerance: float) -> bool:
    if isinstance(want, float) and isinstance(got, float):
        return abs(got - want) <= tolerance
    return bool(got == want)


def differences(
    got: Mapping[str, Result],
    want: Mapping[str, Result],
    tolerance: float = TOLERANCE,
) -> list[str]:
    """Every way got differs from want, one line for people each; none if
    they match. Numbers may differ by tolerance (0.0: not at all)."""
    found = [f"{key}: not expected" for key in got if key not in want]
    for key, expected in want.items():
        result = got.get(key)
        if result is None:
            found.append(f"{key}: missing")
            continue
        for part, label in (("order_id", "order"), ("drivers", "drivers")):
            got_part, want_part = result.get(part), expected.get(part)
            if got_part != want_part:
                found.append(f"{key}: {label} {got_part} for {want_part}")
        scores, wanted = result["scores"] or {}, expected["scores"] or {}
        for name in sorted(set(scores) | set(wanted)):
            value, target = scores.get(name), wanted.get(name)
            if not _same(value, target, tolerance):
                found.append(f"{key}: {name} {value!r} for {target!r}")
    return found
