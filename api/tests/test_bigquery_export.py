"""T306: which Firestore collections stream to BigQuery, and where to.

firebase.json's kit entry has one instance per streamed collection, each
with its settings in bigquery/instances/<instance>/.env (bigquery/src/
index.ts). STREAMED is the table: every collection fitcore.documents knows
is streamed by the instance named here, or deliberately not (None), so a new
collection fails test_streamed_covers_every_collection until its task
decides. bigquery/check.js checks the same files against the kit itself.
"""

import json
from pathlib import Path

import pytest
from fitcore.documents import COLLECTIONS

ROOT = Path(__file__).resolve().parents[2]

# Each collection's kit instance, or None when it is not streamed.
STREAMED: dict[str, str | None] = {
    "events": "events",
    "customers": "customers",
    "orders": "orders",
    "scores": "scores",
    "users": None,  # staff emails stay out of analytics
    "rules": None,  # nothing writes it yet
    "actions": None,  # nothing writes it yet
    "models": None,  # nothing writes it yet
}


def kit_instances() -> dict[str, str]:
    """firebase.json's kit instances: each one's settings directory."""
    config = json.loads((ROOT / "firebase.json").read_text())
    (kit,) = [entry for entry in config["functions"] if "kit" in entry]
    instances: dict[str, str] = kit["instances"]
    return instances


def settings(instance: str) -> dict[str, str]:
    """One instance's .env, as a dict."""
    text = (ROOT / kit_instances()[instance] / ".env").read_text()
    pairs = [line.split("=", 1) for line in text.splitlines() if "=" in line]
    return {key: value for key, value in pairs if not key.startswith("#")}


def test_streamed_covers_every_collection() -> None:
    assert set(STREAMED) == set(COLLECTIONS)


def test_there_is_one_instance_per_streamed_collection() -> None:
    wanted = {instance for instance in STREAMED.values() if instance}
    assert set(kit_instances()) == wanted


@pytest.mark.parametrize(
    "name", [name for name, instance in STREAMED.items() if instance]
)
def test_an_instance_streams_its_collection_to_the_raw_dataset_in_bangkok(
    name: str,
) -> None:
    instance = STREAMED[name]
    assert instance is not None
    found = settings(instance)
    collection = COLLECTIONS[name].path.rsplit("/", 1)[0]
    assert found["COLLECTION_PATH"] == collection
    assert found["TABLE_ID"] == instance
    assert found["WILDCARD_IDS"] == ("true" if "{" in collection else "false")
    assert found["DATASET_ID"] == "raw"
    assert found["DATASET_LOCATION"] == "asia-southeast3"
    assert found["BIGQUERY_PROJECT_ID"] == "cfi-dev-th26"
    assert found["DATABASE_REGION"] == ""  # index.ts sets the functions' region
