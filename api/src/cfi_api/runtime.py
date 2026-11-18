"""What the app runs with, set up once in create_app: the model (from a
file, or from the registry), the store, the publisher and the history."""

from pathlib import Path

from fitcore.model import ModelParams
from flask import current_app

from cfi_api.export import Publisher, PubSubPublisher
from cfi_api.history import BigQueryHistory, History
from cfi_api.model_registry import GcsFiles, LiveModel, MemoryFiles, Models
from cfi_api.store import MemoryStore, Reader, Store

__all__ = ["LiveModel"]


def load_model(path: str | None, version: object) -> LiveModel | None:
    """The model.json at path with its version, or None if neither is set.

    Raises ValueError if only one of the two is set or the file is not a
    valid model.json, and OSError if it cannot be read.
    """
    if path is None and version is None:
        return None
    if path is None or version is None:
        raise ValueError("set MODEL_PATH and MODEL_VERSION together, or neither")
    return LiveModel(ModelParams.from_json(Path(path).read_bytes()), str(version))


def make_store(kind: str, project: str | None, *, outbox: bool = False) -> Store:
    """A MemoryStore for "memory", or a FirestoreStore for "firestore", each
    with an outbox if asked (T306).

    The Firestore client is imported only when it is used: it is slow to
    import, and most tests never need it. Raises ValueError for any other
    kind.
    """
    if kind == "memory":
        return MemoryStore(outbox=outbox)
    if kind == "firestore":
        from google.cloud import firestore

        from cfi_api.firestore_store import FirestoreStore

        return FirestoreStore(firestore.Client(project=project), outbox=outbox)
    raise ValueError(f"STORE must be 'memory' or 'firestore', got {kind!r}")


def make_publisher(topic: str | None, project: str | None) -> Publisher | None:
    """A PubSubPublisher to topic in project, or None without a topic.

    Raises ValueError for a topic without a project.
    """
    if topic is None:
        return None
    if project is None:
        raise ValueError("TOPIC needs PROJECT")
    return PubSubPublisher(project, topic)


def make_models(kind: str | None, project: str | None) -> Models | None:
    """The registry's models (T310): files in Cloud Storage for "gcs", in
    memory for "memory" (tests), or None for None: the model comes from
    MODEL_PATH then. Raises ValueError for any other kind."""
    if kind is None:
        return None
    if kind == "gcs":
        return Models(GcsFiles(project))
    if kind == "memory":
        return Models(MemoryFiles())
    raise ValueError(f"REGISTRY must be 'gcs', 'memory' or unset, got {kind!r}")


def make_history(kind: str | None, project: str | None) -> History | None:
    """A BigQueryHistory in project for "bigquery", or None for None.

    The client is imported only when it is used, as Firestore's is. Raises
    ValueError for any other kind.
    """
    if kind is None:
        return None
    if kind == "bigquery":
        from google.cloud import bigquery

        return BigQueryHistory(bigquery.Client(project=project))
    raise ValueError(f"HISTORY must be 'bigquery' or unset, got {kind!r}")


def live_model(db: Reader | None = None) -> LiveModel | None:
    """The model to score with, or None if there is none.

    Without a registry: MODEL_PATH's. With one: its live version, read
    through db (the store, if not given), so an event reads it inside its
    own unit of work. Raises BadModel if that version's file will not load.
    """
    models: Models | None = current_app.extensions["cfi_models"]
    if models is None:
        model: LiveModel | None = current_app.extensions["cfi_model"]
        return model
    entry = (db if db is not None else store()).find_live_model()
    return None if entry is None else models.load(entry)


def live_version() -> str | None:
    """The live model's version, without loading it."""
    models: Models | None = current_app.extensions["cfi_models"]
    if models is None:
        model: LiveModel | None = current_app.extensions["cfi_model"]
        return None if model is None else model.version
    entry = store().find_live_model()
    return None if entry is None else entry.version


def registry() -> Models | None:
    """The current app's registry models, or None without a registry."""
    models: Models | None = current_app.extensions["cfi_models"]
    return models


def store() -> Store:
    """The current app's store."""
    current: Store = current_app.extensions["cfi_store"]
    return current


def publisher() -> Publisher | None:
    """The current app's publisher, or None when it exports nothing."""
    current: Publisher | None = current_app.extensions["cfi_publisher"]
    return current


def history() -> History | None:
    """The current app's history, or None when it reads no BigQuery."""
    current: History | None = current_app.extensions["cfi_history"]
    return current
