"""The model registry (T310): which model is live, and its files.

Each version's model.json and model_card.md sit in Cloud Storage, at
gs://{bucket}/models/{version}/, and models/{version} in the store records
it: status (candidate, live or retired; one live at most), metrics, and the
SHA-256 of its model.json. Promoting a version makes it live and retires
the one before; promoting an older version rolls back.

Every event reads the live version inside its own transaction (Reader.
find_live_model), so on every instance an event is scored by the model that
was live when it committed: a promote changes the document that event read,
so Firestore runs the event again. Each instance downloads a version's
model.json once, checks its SHA-256, and keeps it (Models).

Admins register and promote versions from the command line:

    python -m cfi_api.registry --project P register VERSION MODEL_JSON \\
        --card CARD --bucket BUCKET --trained YYYY-MM-DD --metric churn_auc=0.789
    python -m cfi_api.registry --project P list
    python -m cfi_api.registry --project P promote VERSION

It runs as you, with Application Default Credentials, as cfi_api.roles
does. The API answers GET /models and POST /models/{version}/promote too.
"""

import argparse
import hashlib
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Protocol

from fitcore.documents import check
from fitcore.model import ModelParams

LIVE = "live"


@dataclass(frozen=True)
class LiveModel:
    """The model the API scores with, and the version it reports."""

    params: ModelParams
    version: str


@dataclass(frozen=True)
class ModelEntry:
    """models/{version}: one version in the registry."""

    version: str
    status: str
    path: str  # gs://{bucket}/models/{version}/model.json
    sha256: str
    metrics: Mapping[str, float]
    created_at: datetime
    promoted_at: datetime | None


def model_document(entry: ModelEntry) -> dict[str, Any]:
    """The document for models/{version}, checked against its schema."""
    document = {
        "version": entry.version,
        "status": entry.status,
        "path": entry.path,
        "sha256": entry.sha256,
        "metrics": dict(entry.metrics),
        "created_at": entry.created_at,
        "promoted_at": entry.promoted_at,
    }
    check("models", document)
    return document


def entry_from_document(document: Mapping[str, Any]) -> ModelEntry:
    return ModelEntry(
        version=document["version"],
        status=document["status"],
        path=document["path"],
        sha256=document["sha256"],
        metrics=dict(document["metrics"]),
        created_at=document["created_at"],
        promoted_at=document["promoted_at"],
    )


def model_path(bucket: str, version: str, name: str = "model.json") -> str:
    """Where a version's files sit: gs://{bucket}/models/{version}/{name}."""
    return f"gs://{bucket}/models/{version}/{name}"


def card_path(path: str) -> str:
    """The model_card.md next to a model.json."""
    return path.rsplit("/", 1)[0] + "/model_card.md"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UnknownVersion(KeyError):
    """No such version in the registry."""


class BadModel(ValueError):
    """A version's model.json is missing, does not match its SHA-256, or is
    not a valid model.json."""


# --- Files ---


class Files(Protocol):
    """Where model files are kept, by gs:// path."""

    def read(self, path: str) -> bytes: ...

    def write(self, path: str, data: bytes, content_type: str) -> None: ...


def _split(path: str) -> tuple[str, str]:
    if not path.startswith("gs://") or "/" not in path[5:]:
        raise ValueError(f"not a gs:// path to a file: {path!r}")
    bucket, name = path[5:].split("/", 1)
    return bucket, name


class GcsFiles:
    """Files in Cloud Storage. The client is made on first use: it needs
    credentials, which tests and local runs without a registry lack."""

    def __init__(self, project: str | None) -> None:
        self._project = project
        self._client: Any = None
        self._lock = threading.Lock()

    def _bucket(self, name: str) -> Any:
        with self._lock:
            if self._client is None:
                from google.cloud.storage import Client

                self._client = Client(project=self._project)
        return self._client.bucket(name)

    def read(self, path: str) -> bytes:
        bucket, name = _split(path)
        data: bytes = self._bucket(bucket).blob(name).download_as_bytes()
        return data

    def write(self, path: str, data: bytes, content_type: str) -> None:
        bucket, name = _split(path)
        self._bucket(bucket).blob(name).upload_from_string(
            data, content_type=content_type
        )


class MemoryFiles:
    """Files in this process's memory, for tests and local runs."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.reads = 0

    def read(self, path: str) -> bytes:
        _split(path)
        self.reads += 1
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    def write(self, path: str, data: bytes, content_type: str) -> None:
        _split(path)
        self.files[path] = data


# --- Models: each version's model, loaded once ---


class Models:
    """The registry's models as this instance has loaded them, by version."""

    def __init__(self, files: Files) -> None:
        self.files = files
        self._loaded: dict[tuple[str, str], LiveModel] = {}
        self._lock = threading.Lock()

    def load(self, entry: ModelEntry) -> LiveModel:
        """entry's model: downloaded and checked the first time, then kept.

        Raises BadModel if its file is missing, does not match the entry's
        SHA-256, or is not a valid model.json.
        """
        key = (entry.version, entry.sha256)
        with self._lock:
            found = self._loaded.get(key)
        if found is not None:
            return found
        try:
            data = self.files.read(entry.path)
        except FileNotFoundError as err:
            raise BadModel(f"{entry.version}: {entry.path} is missing") from err
        if sha256(data) != entry.sha256:
            raise BadModel(f"{entry.version}: {entry.path} does not match its SHA-256")
        try:
            params = ModelParams.from_json(data)
        except (ValueError, TypeError) as err:
            raise BadModel(f"{entry.version}: {err}") from err
        model = LiveModel(params, entry.version)
        with self._lock:
            self._loaded[key] = model
        return model


# --- Admin commands ---


class Registry(Protocol):
    """What the admin commands need from a store."""

    def list_models(self) -> list[ModelEntry]: ...

    def add_model(self, entry: ModelEntry) -> None: ...

    def promote_model(self, version: str, now: datetime) -> ModelEntry: ...


def register(
    store: Registry,
    files: Files,
    *,
    version: str,
    model_json: bytes,
    card: bytes,
    bucket: str,
    trained: datetime,
    metrics: Mapping[str, float],
) -> ModelEntry:
    """Upload a version's files and add it as a candidate.

    Raises ValueError for a version already registered, and BadModel for a
    model.json that fitcore.model would refuse. Nothing is uploaded then.
    """
    if any(entry.version == version for entry in store.list_models()):
        raise ValueError(f"{version} is already registered")
    try:
        ModelParams.from_json(model_json)
    except (ValueError, TypeError) as err:
        raise BadModel(f"{version}: {err}") from err
    path = model_path(bucket, version)
    entry = ModelEntry(
        version=version,
        status="candidate",
        path=path,
        sha256=sha256(model_json),
        metrics=dict(metrics),
        created_at=trained,
        promoted_at=None,
    )
    model_document(entry)  # checks it before anything is written
    files.write(path, model_json, "application/json")
    files.write(card_path(path), card, "text/markdown")
    store.add_model(entry)
    return entry


def promote(store: Registry, models: Models, version: str, now: datetime) -> ModelEntry:
    """Make version live, after checking its file loads; the live one before
    it is retired. Promoting the live version changes nothing.

    Raises UnknownVersion, or BadModel for a file that would not load: the
    live version stays live then.
    """
    found = {entry.version: entry for entry in store.list_models()}
    if version not in found:
        raise UnknownVersion(version)
    models.load(found[version])
    return store.promote_model(version, now)


def _metric(text: str) -> tuple[str, float]:
    name, _, value = text.partition("=")
    if not name or not value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name, float(value)


def main(
    argv: Sequence[str] | None = None,
    *,
    store: Registry | None = None,
    files: Files | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfi_api.registry", description="The model registry (T310)."
    )
    parser.add_argument("--project", help="the Google Cloud project")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("register", help="upload a version as a candidate")
    add.add_argument("version")
    add.add_argument("model_json", type=Path)
    add.add_argument("--card", type=Path, required=True, help="its model_card.md")
    add.add_argument("--bucket", required=True, help="the models bucket")
    add.add_argument("--trained", type=date.fromisoformat, required=True)
    add.add_argument("--metric", type=_metric, action="append", default=[])
    commands.add_parser("list", help="every version, newest first")
    up = commands.add_parser("promote", help="make a version live")
    up.add_argument("version")
    args = parser.parse_args(argv)

    if store is None or files is None:
        if not args.project:
            print("Give --project, the project with the registry.", file=sys.stderr)
            return 2
        from google.cloud import firestore

        from cfi_api.firestore_store import FirestoreStore

        if store is None:
            store = FirestoreStore(firestore.Client(project=args.project))
        if files is None:
            files = GcsFiles(args.project)

    try:
        if args.command == "register":
            entry = register(
                store,
                files,
                version=args.version,
                model_json=args.model_json.read_bytes(),
                card=args.card.read_bytes(),
                bucket=args.bucket,
                trained=datetime.combine(args.trained, time(), UTC),
                metrics=dict(args.metric),
            )
            print(f"Registered {entry.version} as a candidate: {entry.path}")
        elif args.command == "promote":
            entry = promote(store, Models(files), args.version, datetime.now(UTC))
            print(f"{entry.version} is live.")
        for entry in store.list_models():
            print(f"  {entry.version:20} {entry.status:9} {entry.path}")
    except (ValueError, KeyError) as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
