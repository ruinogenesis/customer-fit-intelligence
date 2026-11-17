"""T309: the golden replay through the API, with each store.

POST /events gets the frozen log one event at a time, and its answers must
match fitcore's replay of the same log exactly and the golden file within
its tolerance (fitcore.golden). The Firestore store runs on the emulator
only: there, each event's customer and order are saved and read back before
the next event, so the test also proves they survive Firestore.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from fitcore.golden import Result, differences, read_events, read_golden, replay
from fitcore.model import ModelParams

from cfi_api import create_app

RESULTS = Path(__file__).resolve().parents[2] / "notebooks/results"
EVENTS = RESULTS / "t309_events.jsonl"
GOLDEN = RESULTS / "t309_golden.json"

needs_emulator = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)


@pytest.fixture(scope="module")
def events() -> list[dict[str, Any]]:
    return read_events(EVENTS)


@pytest.mark.parametrize(
    "store", ["memory", pytest.param("firestore", marks=needs_emulator)]
)
def test_the_api_scores_the_log_as_fitcore_and_the_golden_file_do(
    store: str, reference_model: str, events: list[dict[str, Any]]
) -> None:
    config = {"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"}
    config |= {"STORE": store, "PROJECT": "demo-cfi"}
    client = create_app(config).test_client()
    got: dict[str, Result] = {}
    for event in events:
        response = client.post("/api/v1/events", json=event)
        # 200: already received, in an emulator that outlived an earlier run
        assert response.status_code in (200, 201), response.get_json()
        answer = response.get_json()
        got[event["event_id"]] = {k: answer[k] for k in ("order_id", "scores")}
    fitcore = replay(ModelParams.from_json(Path(reference_model).read_bytes()), events)
    assert differences(got, fitcore, tolerance=0.0) == []
    assert differences(got, read_golden(GOLDEN).results) == []
