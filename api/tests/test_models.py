"""T310: the model registry. Promote and rollback switch the live model with
no redeploy: the next event is scored by whatever is live, on every instance,
because each event reads the live version in its own unit of work.

These run on the memory store with the files in memory (REGISTRY "memory");
test_firestore_store.py runs the store's half against the Firestore emulator.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient
from test_access import bearer
from test_events import delivered, placed

from cfi_api import create_app, registry
from cfi_api.access import ACCESS, EMULATOR
from cfi_api.model_registry import (
    BadModel,
    GcsFiles,
    MemoryFiles,
    ModelEntry,
    Models,
    card_path,
    model_path,
    register,
    sha256,
)
from cfi_api.store import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
T112 = (ROOT / "notebooks/results/t112_model.json").read_bytes()
BUCKET = "cfi-test-models"
TRAINED = datetime(2026, 11, 14, tzinfo=UTC)
API = "/api/v1"


def other_model(kappa: float = 2.0) -> bytes:
    """T112's model with a different kappa: same shape, different scores."""
    document = json.loads(T112)
    document["kappa"] = [kappa] * len(document["kappa"])
    return json.dumps(document).encode()


def _app(**config: Any) -> Flask:
    with pytest.MonkeyPatch.context() as env:
        for name in [name for name in os.environ if name.startswith("CFI_")]:
            env.delenv(name)
        return create_app({"REGISTRY": "memory", **config})


@pytest.fixture
def app() -> Flask:
    return _app()


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def files(app: Flask) -> MemoryFiles:
    models = app.extensions["cfi_models"]
    assert isinstance(models, Models) and isinstance(models.files, MemoryFiles)
    return models.files


def add(app: Flask, version: str, data: bytes = T112) -> ModelEntry:
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    return register(
        store,
        files(app),
        version=version,
        model_json=data,
        card=b"# Model card\n",
        bucket=BUCKET,
        trained=TRAINED,
        metrics={"churn_auc": 0.789},
    )


def promote(client: FlaskClient, version: str) -> Any:
    return client.post(f"{API}/models/{version}/promote")


def statuses(client: FlaskClient) -> dict[str, str]:
    answer = client.get(f"{API}/models")
    assert answer.status_code == 200
    return {m["version"]: m["status"] for m in answer.get_json()["models"]}


def late(client: FlaskClient, path: str = "/events") -> Any:
    """A what-if (POST /score) or a real late delivery, after its order."""
    client.post(f"{API}/events", json=placed())
    return client.post(f"{API}{path}", json=delivered("ord_1", 45)).get_json()


# --- Registering ---


def test_register_uploads_both_files_and_adds_a_candidate(
    app: Flask, client: FlaskClient
) -> None:
    entry = add(app, "t112-sim")
    path = model_path(BUCKET, "t112-sim")
    assert entry.path == path == "gs://cfi-test-models/models/t112-sim/model.json"
    assert (entry.status, entry.sha256, entry.promoted_at) == (
        "candidate",
        sha256(T112),
        None,
    )
    assert files(app).files == {path: T112, card_path(path): b"# Model card\n"}
    listed = client.get(f"{API}/models").get_json()["models"]
    assert [m["version"] for m in listed] == ["t112-sim"]
    assert listed[0]["metrics"] == {"churn_auc": 0.789}


def test_a_version_registers_once_and_must_be_a_model(app: Flask) -> None:
    add(app, "t112-sim")
    with pytest.raises(ValueError, match="already registered"):
        add(app, "t112-sim", other_model())
    with pytest.raises(BadModel):
        add(app, "broken", b'{"format": 1}')
    assert all("broken" not in path for path in files(app).files)


# --- Promoting and rolling back ---


def test_without_a_live_version_events_are_503(client: FlaskClient) -> None:
    assert client.post(f"{API}/events", json=placed()).status_code == 503
    assert client.get(f"{API}/health").get_json()["model_version"] is None


def test_promote_makes_a_version_live_for_the_next_event(
    app: Flask, client: FlaskClient
) -> None:
    add(app, "t112-sim")
    answer = promote(client, "t112-sim")
    assert answer.status_code == 200
    assert answer.get_json()["status"] == "live"
    assert answer.get_json()["promoted_at"] is not None
    assert client.get(f"{API}/health").get_json()["model_version"] == "t112-sim"
    assert late(client)["model_version"] == "t112-sim"


def test_promote_and_rollback_switch_the_model_with_no_redeploy(
    app: Flask, client: FlaskClient
) -> None:
    add(app, "t112-sim")
    add(app, "t310-other", other_model())
    promote(client, "t112-sim")
    before = late(client, "/score")
    assert promote(client, "t310-other").status_code == 200
    assert statuses(client) == {"t112-sim": "retired", "t310-other": "live"}
    other = client.post(f"{API}/score", json=delivered("ord_1", 45)).get_json()
    assert other["model_version"] == "t310-other"
    assert other["scores"]["fit"] != before["scores"]["fit"]
    assert promote(client, "t112-sim").status_code == 200  # rolling back
    assert statuses(client) == {"t112-sim": "live", "t310-other": "retired"}
    again = client.post(f"{API}/score", json=delivered("ord_1", 45)).get_json()
    assert (again["model_version"], again["scores"]) == ("t112-sim", before["scores"])


def test_promoting_the_live_version_changes_nothing(
    app: Flask, client: FlaskClient
) -> None:
    add(app, "t112-sim")
    first = promote(client, "t112-sim").get_json()
    second = promote(client, "t112-sim")
    assert second.status_code == 200
    assert second.get_json() == first


def test_an_unknown_version_is_404(client: FlaskClient) -> None:
    assert promote(client, "t999").status_code == 404


def test_a_file_that_does_not_match_its_sha256_is_not_promoted(
    app: Flask, client: FlaskClient
) -> None:
    add(app, "t112-sim")
    promote(client, "t112-sim")
    entry = add(app, "t310-other", other_model())
    files(app).files[entry.path] = other_model(kappa=3.0)  # changed after register
    answer = promote(client, "t310-other")
    assert answer.status_code == 409
    assert "SHA-256" in answer.get_json()["message"]
    assert statuses(client) == {"t112-sim": "live", "t310-other": "candidate"}


def test_a_live_file_that_will_not_load_makes_events_503(
    app: Flask, client: FlaskClient
) -> None:
    entry = add(app, "t112-sim")
    promote(client, "t112-sim")
    files(app).files[entry.path] = b"{}"
    # A new instance has loaded nothing yet, so it reads the file.
    app.extensions["cfi_models"] = Models(files(app))
    answer = client.post(f"{API}/events", json=placed())
    assert answer.status_code == 503
    assert "cannot be loaded" in answer.get_json()["message"]


def test_each_instance_downloads_a_version_once(
    app: Flask, client: FlaskClient
) -> None:
    add(app, "t112-sim")
    promote(client, "t112-sim")
    reads = files(app).reads
    for minutes in (45, 50, 55):
        client.post(f"{API}/score", json=delivered("ord_1", minutes))
    late(client)
    assert files(app).reads == reads


# --- Settings, access and docs ---


def test_without_a_registry_models_are_503(reference_model: str) -> None:
    client = _app(
        REGISTRY=None, MODEL_PATH=reference_model, MODEL_VERSION="t112-sim"
    ).test_client()
    assert client.get(f"{API}/models").status_code == 503
    assert promote(client, "t112-sim").status_code == 503


def test_a_model_file_and_a_registry_together_are_refused(
    reference_model: str,
) -> None:
    with pytest.raises(ValueError, match="not both"):
        _app(MODEL_PATH=reference_model, MODEL_VERSION="t112-sim")
    with pytest.raises(ValueError, match="REGISTRY"):
        _app(REGISTRY="dropbox")


def test_staff_see_the_models_and_only_admins_promote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EMULATOR, "127.0.0.1:9099")
    app = _app(AUTH="firebase", PROJECT="demo-cfi")
    add(app, "t112-sim")
    client = app.test_client()
    assert client.get(f"{API}/models").status_code == 401
    assert client.get(f"{API}/models", headers=bearer("viewer")).status_code == 200
    for role in ("viewer", "analyst"):
        refused = client.post(f"{API}/models/t112-sim/promote", headers=bearer(role))
        assert refused.status_code == 403
    done = client.post(f"{API}/models/t112-sim/promote", headers=bearer("admin"))
    assert done.status_code == 200
    assert ACCESS["models.Promote", "POST"].role == "admin"
    card = f"{API}/models/t112-sim/card"
    assert client.get(card).status_code == 401
    assert client.get(card, headers=bearer("viewer")).status_code == 200


def test_the_endpoints_are_documented(client: FlaskClient) -> None:
    paths = client.get(f"{API}/openapi.json").get_json()["paths"]
    listing = paths[f"{API}/models"]["get"]["responses"]
    promoting = paths[f"{API}/models/{{version}}/promote"]["post"]["responses"]
    carding = paths[f"{API}/models/{{version}}/card"]["get"]["responses"]
    assert {"200", "401", "403", "503"} <= set(listing)
    assert {"200", "401", "403", "404", "409", "503"} <= set(promoting)
    assert {"200", "401", "403", "404", "503"} <= set(carding)
    assert "text/markdown" in carding["200"]["content"]


# --- Model cards (T407) ---


def test_staff_read_a_versions_model_card(app: Flask, client: FlaskClient) -> None:
    add(app, "t112-sim")
    answer = client.get(f"{API}/models/t112-sim/card")
    assert answer.status_code == 200
    assert answer.mimetype == "text/markdown"
    assert answer.get_data(as_text=True) == "# Model card\n"


def test_a_card_for_an_unknown_version_or_a_missing_file_is_404(
    app: Flask, client: FlaskClient
) -> None:
    entry = add(app, "t112-sim")
    assert client.get(f"{API}/models/t999/card").status_code == 404
    del files(app).files[card_path(entry.path)]
    answer = client.get(f"{API}/models/t112-sim/card")
    assert answer.status_code == 404
    assert "no model card" in answer.get_json()["message"]


def test_without_a_registry_cards_are_503(reference_model: str) -> None:
    app = _app(REGISTRY=None, MODEL_PATH=reference_model, MODEL_VERSION="t112-sim")
    assert app.test_client().get(f"{API}/models/t112-sim/card").status_code == 503


def test_a_missing_cloud_storage_file_is_file_not_found() -> None:
    from google.api_core.exceptions import NotFound

    class Blob:
        def download_as_bytes(self) -> bytes:
            raise NotFound("No such object")

    class Bucket:
        def blob(self, name: str) -> Blob:
            return Blob()

    class Client:
        def bucket(self, name: str) -> Bucket:
            return Bucket()

    gcs = GcsFiles(project=None)
    gcs._client = Client()
    with pytest.raises(FileNotFoundError, match="model_card.md"):
        gcs.read("gs://cfi-test-models/models/t112-sim/model_card.md")


# --- The command line ---


def test_the_command_line_registers_lists_and_promotes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store, memory = MemoryStore(), MemoryFiles()
    (tmp_path / "model.json").write_bytes(T112)
    (tmp_path / "card.md").write_text("# Model card\n")
    argv = [
        "register",
        "t112-sim",
        str(tmp_path / "model.json"),
        "--card",
        str(tmp_path / "card.md"),
        "--bucket",
        BUCKET,
        "--trained",
        "2026-11-14",
        "--metric",
        "churn_auc=0.789",
        "--metric",
        "reorder_auc=0.711",
    ]
    assert registry.main(argv, store=store, files=memory) == 0
    entry = store.list_models()[0]
    assert (entry.status, entry.created_at) == ("candidate", TRAINED)
    assert entry.metrics == {"churn_auc": 0.789, "reorder_auc": 0.711}
    assert registry.main(["promote", "t112-sim"], store=store, files=memory) == 0
    assert store.find_live_model() == store.list_models()[0]
    assert registry.main(["list"], store=store, files=memory) == 0
    out = capsys.readouterr().out
    assert "Registered t112-sim as a candidate" in out and "t112-sim is live." in out
    assert registry.main(argv, store=store, files=memory) == 1  # twice
    assert registry.main(["promote", "t999"], store=store, files=memory) == 1
    assert "Refused" in capsys.readouterr().err


def test_the_command_line_needs_a_project(capsys: pytest.CaptureFixture[str]) -> None:
    assert registry.main(["list"]) == 2
    assert "--project" in capsys.readouterr().err


def test_the_command_line_runs_without_a_double_import_warning() -> None:
    # cfi_api imports the library (model_registry), never the command line,
    # so python -m cfi_api.registry loads it once.
    command = ["-W", "error::RuntimeWarning", "-m", "cfi_api.registry", "list"]
    run = subprocess.run(
        [sys.executable, *command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (run.returncode, run.stderr.strip()) == (
        2,
        "Give --project, the project with the registry.",
    )
