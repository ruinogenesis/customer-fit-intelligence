"""T317: Firestore indexes score history on `at` alone.

/customers/{id}/scores is the only query on score entries, and it orders by
SCORES_ORDER. Every other field of an entry, the 48 contributions above all,
is left out of the indexes in firestore.indexes.json (about 1.2 KB stored per
entry instead of 19 KB). Questions across scorings go to BigQuery's
raw.scores (T306).
"""

import json
from pathlib import Path

import pytest
from flask import Flask
from test_events import URL, memory, placed

from cfi_api import create_app
from cfi_api.firestore_store import SCORES_ORDER

INDEXES = Path(__file__).parents[2] / "firestore.indexes.json"


@pytest.fixture
def app(reference_model: str) -> Flask:
    return create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})


def test_score_history_is_indexed_only_on_the_field_it_is_ordered_by(
    app: Flask,
) -> None:
    assert app.test_client().post(URL, json=placed()).status_code == 201
    [entry] = [
        document
        for path, document in memory(app).documents.items()
        if path.startswith("customers/cust_1/scores/")
    ]
    overrides = [
        override
        for override in json.loads(INDEXES.read_text())["fieldOverrides"]
        if override["collectionGroup"] == "scores"
    ]
    # Every field an entry holds has a decision: exempt, or the one queried.
    exempt = sorted(override["fieldPath"] for override in overrides)
    assert SCORES_ORDER in entry
    assert exempt == sorted(set(entry) - {SCORES_ORDER})
    assert all(override["indexes"] == [] for override in overrides)
    assert not any("ttl" in override for override in overrides)
