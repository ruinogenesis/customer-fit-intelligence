"""What the app runs with, set up once in create_app: the model, the store
and the publisher."""

from dataclasses import dataclass
from pathlib import Path

from fitcore.model import ModelParams
from flask import current_app

from cfi_api.export import Publisher, PubSubPublisher
from cfi_api.store import MemoryStore, Store


@dataclass(frozen=True)
class LiveModel:
    """The model the API scores with, and the version it reports."""

    params: ModelParams
    version: str


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


def live_model() -> LiveModel | None:
    """The current app's model, or None when it has none."""
    model: LiveModel | None = current_app.extensions["cfi_model"]
    return model


def store() -> Store:
    """The current app's store."""
    current: Store = current_app.extensions["cfi_store"]
    return current


def publisher() -> Publisher | None:
    """The current app's publisher, or None when it exports nothing."""
    current: Publisher | None = current_app.extensions["cfi_publisher"]
    return current
