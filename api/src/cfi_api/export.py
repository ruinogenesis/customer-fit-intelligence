"""T306: what POST /events saves goes on to BigQuery, through Pub/Sub.

Every document an event saves becomes one Pub/Sub message: a row of the
BigQuery table raw.changes (bigquery/raw.sql), which a BigQuery subscription
writes. With TOPIC set, the store saves an event's messages in an outbox
entry in the event's own transaction, so they exist exactly when the event
does.

After the commit, POST /events publishes them and deletes the entry. If
Pub/Sub fails, the event is still answered and its entry stays: a later
POST /events sends leftovers over a minute old, up to ten at a time. So a
message can reach BigQuery twice; the views in bigquery/raw.sql keep one
row per (event_id, path). Without TOPIC nothing is exported.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from fitcore.documents import check

logger = logging.getLogger(__name__)

# An outbox entry this old is a leftover: its own request has given up on it.
LEFTOVER_AGE = timedelta(minutes=1)
# The most leftovers one request sends, so a backlog never holds one up long.
LEFTOVERS_PER_REQUEST = 10
# Seconds to wait for Pub/Sub to take one event's messages.
PUBLISH_TIMEOUT = 5


@dataclass(frozen=True)
class OutboxEntry:
    """One event's messages, kept in the outbox until Pub/Sub has them all."""

    event_id: str
    messages: tuple[str, ...]
    created_at: datetime


def _json(value: object) -> str:
    """A timestamp as JSON: ISO 8601 in UTC, as Firestore keeps it."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON")


def message(
    path: str, document: Mapping[str, Any], event_id: str, written_at: datetime
) -> str:
    """One saved document as a row of raw.changes, in JSON."""
    return json.dumps(
        {
            "collection": path.split("/")[-2],
            "path": path,
            "event_id": event_id,
            "written_at": _json(written_at),
            # BigQuery takes a JSON column as JSON text inside the message.
            "document": json.dumps(document, default=_json, sort_keys=True),
        },
        sort_keys=True,
    )


def outbox_entry(
    event_id: str,
    documents: Sequence[tuple[str, Mapping[str, Any]]],
    written_at: datetime,
) -> OutboxEntry:
    """The entry for an event that saved documents, (path, document), at written_at."""
    return OutboxEntry(
        event_id,
        tuple(message(path, doc, event_id, written_at) for path, doc in documents),
        written_at,
    )


def outbox_document(entry: OutboxEntry) -> dict[str, Any]:
    """entry as its outbox/{event_id} document, checked."""
    document = {
        "event_id": entry.event_id,
        "messages": list(entry.messages),
        "created_at": entry.created_at,
    }
    check("outbox", document)
    return document


def outbox_from_document(document: Mapping[str, Any]) -> OutboxEntry:
    return OutboxEntry(
        document["event_id"], tuple(document["messages"]), document["created_at"]
    )


class Outbox(Protocol):
    """Where a store keeps outbox entries."""

    def unsent(self, before: datetime, limit: int) -> list[OutboxEntry]:
        """Up to limit entries created before `before`, oldest first."""
        ...

    def sent(self, event_id: str) -> None:
        """Delete event_id's entry: Pub/Sub has every message in it."""
        ...


class Publisher(Protocol):
    def publish(self, messages: Sequence[str]) -> None:
        """Send messages to the topic; raise unless Pub/Sub took them all."""
        ...


class PubSubPublisher:
    """A Publisher to a Pub/Sub topic, or to the emulator at PUBSUB_EMULATOR_HOST."""

    def __init__(self, project: str, topic: str) -> None:
        # Imported only when used: it is slow to import.
        from google.cloud.pubsub_v1 import PublisherClient

        self._client = PublisherClient()
        self._topic = self._client.topic_path(project, topic)

    def publish(self, messages: Sequence[str]) -> None:
        futures = [self._client.publish(self._topic, m.encode()) for m in messages]
        for future in futures:
            future.result(timeout=PUBLISH_TIMEOUT)


class MemoryPublisher:
    """A Publisher for tests: it keeps what it is sent, and fails while failing."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.failing = False

    def publish(self, messages: Sequence[str]) -> None:
        if self.failing:
            raise ConnectionError("Pub/Sub is unreachable")
        self.sent.extend(messages)


def export(
    outbox: Outbox, publisher: Publisher, entry: OutboxEntry | None, now: datetime
) -> None:
    """Publish entry, then up to LEFTOVERS_PER_REQUEST leftovers, deleting each
    one Pub/Sub took.

    Stops at the first failure and logs it, but never raises: the event is
    saved either way, and what was not sent stays in the outbox.
    """
    try:
        if entry is not None:
            publisher.publish(entry.messages)
            outbox.sent(entry.event_id)
        for leftover in outbox.unsent(now - LEFTOVER_AGE, LEFTOVERS_PER_REQUEST):
            publisher.publish(leftover.messages)
            outbox.sent(leftover.event_id)
    except Exception:
        logger.exception("Export to Pub/Sub stopped; the rest stays in the outbox")
