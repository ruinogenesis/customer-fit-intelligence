"""Where the API keeps events, customers, orders and score history.

Store is what the events endpoint needs, and Reader is its reading half:
all that what-if scoring gets. MemoryStore keeps everything in this process
(tests, and trying the API without Firestore);
cfi_api.firestore_store.FirestoreStore keeps it in Firestore. One event is
saved with one call, so a store can write it all at once.

writes() lists the Firestore documents one event saves, each checked
against its schema (fitcore.documents, T301). Both stores call it, so a
document of the wrong shape fails the request in tests as it would in
Firestore.

handle() runs one event's reads, its scoring and its writes as one unit
(T302): no other event's change can land between them, so two events for
one customer at the same moment are applied one after the other instead of
the second erasing the first.

A store made with outbox=True also saves the event's outbox entry in that
unit, for cfi_api.export to send on to BigQuery (T306).

Browser is what the dashboard's read endpoints need (T311): one customer,
pages of customers, of a customer's scores and of their orders, and the
customers per quadrant. Both stores answer from the same documents, so the
API answers the same from either.

The model registry lives in the store too (T310, cfi_api.model_registry): a
Reader finds the live version, inside an event's own unit of work, and a
Store lists, adds and promotes versions.
"""

import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, NamedTuple, Protocol

from fitcore.documents import check, customer_document, order_document
from fitcore.model import LATENTS, QUADRANTS
from fitcore.scoring import Applied, Customer, Explanation, Order, Scores

from cfi_api.export import OutboxEntry, outbox_document, outbox_entry
from cfi_api.model_registry import (
    LIVE,
    ModelEntry,
    UnknownVersion,
    entry_from_document,
    model_document,
)

# Where a list resumes: the sort key of the previous page's last document.
# Customers go riskiest first, then by id; scores and orders newest first.
CustomerKey = tuple[float | None, str]  # churn_risk (None: not scored), id
TimeKey = tuple[datetime, str]  # at or placed_at, then the event or order id


@dataclass(frozen=True)
class EventRecord:
    """An event as received (validated, event_id as text) and the result it got."""

    event: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class HistoryEntry:
    """One scoring of one order, kept in its customer's score history."""

    event_id: str
    order_id: str
    at: datetime
    scores: Scores
    explained: Explanation
    model_version: str


def history_entry(
    record: EventRecord, applied: Applied, model_version: str
) -> HistoryEntry | None:
    """The history entry for an event that scored an order, else None."""
    if applied.scored is None or applied.explained is None or applied.order is None:
        return None
    return HistoryEntry(
        event_id=record.event["event_id"],
        order_id=applied.order.order_id,
        at=record.event["occurred_at"],
        scores=applied.scored,
        explained=applied.explained,
        model_version=model_version,
    )


def history_document(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "event_id": entry.event_id,
        "order_id": entry.order_id,
        "at": entry.at,
        "scores": {name: getattr(entry.scores, name) for name in LATENTS},
        "churn_risk": entry.scores.churn_risk,
        "quadrant": entry.scores.quadrant,
        "contributions": {
            name: dict(shares) for name, shares in entry.explained.contributions.items()
        },
        "drivers": list(entry.explained.drivers),
        "model_version": entry.model_version,
    }


def writes(
    record: EventRecord, applied: Applied, model_version: str
) -> list[tuple[str, dict[str, Any]]]:
    """The documents one event saves, as (path, document), each checked.

    Raises fitcore.documents.DocumentError if one does not fit its schema.
    """
    customer_id = applied.customer.customer_id
    documents = [
        (
            "events",
            f"events/{record.event['event_id']}",
            {"event": record.event, "result": record.result},
        ),
        (
            "customers",
            f"customers/{customer_id}",
            customer_document(applied.customer, model_version),
        ),
    ]
    if applied.order is not None:
        path = f"orders/{applied.order.order_id}"
        documents.append(("orders", path, order_document(applied.order)))
    entry = history_entry(record, applied, model_version)
    if entry is not None:
        path = f"customers/{customer_id}/scores/{entry.event_id}"
        documents.append(("scores", path, history_document(entry)))
    for collection, _, document in documents:
        check(collection, document)
    return [(path, document) for _, path, document in documents]


class Reader(Protocol):
    """The reading half of a Store.

    Code given a Reader cannot save: mypy refuses the call. That is how
    POST /score stays write-free.
    """

    def find_event(self, event_id: str) -> EventRecord | None: ...

    def find_customer(self, customer_id: str) -> Customer | None: ...

    def find_order(self, order_id: str) -> Order | None: ...

    def find_live_model(self) -> ModelEntry | None:
        """The registry's live version, or None if no version is live (T310)."""
        ...


# One event's work: read what it needs through the Reader, and answer the
# result and what changed (None: nothing to save). It must only read, as a
# store may run it again.
Work = Callable[[Reader], tuple[dict[str, Any], Applied | None]]


class Handled(NamedTuple):
    """What handle() returns: work's result and change, and the outbox entry
    it saved (None if it saved nothing or the store has no outbox)."""

    result: dict[str, Any]
    applied: Applied | None
    outbox: OutboxEntry | None


@dataclass(frozen=True)
class Page:
    """One page of a list: its documents, and the key the next page starts
    after (None when this page is the last)."""

    documents: list[dict[str, Any]]
    next: CustomerKey | TimeKey | None


class Browser(Protocol):
    """What the dashboard's read endpoints need (T311), from stored documents."""

    def customer_document(self, customer_id: str) -> dict[str, Any] | None:
        """customers/{customer_id}, or None if there is no such customer."""
        ...

    def list_customers(
        self,
        *,
        quadrant: str | None,
        min_churn_risk: float | None,
        limit: int,
        after: CustomerKey | None,
    ) -> Page:
        """Customers riskiest first (unscored last), then by id, high to low.

        Only those in quadrant and at or above min_churn_risk, when given.
        """
        ...

    def list_scores(
        self, customer_id: str, *, limit: int, after: TimeKey | None
    ) -> Page:
        """A customer's score history, newest first."""
        ...

    def list_orders(
        self, customer_id: str, *, limit: int, after: TimeKey | None
    ) -> Page:
        """A customer's orders, newest first."""
        ...

    def count_quadrants(self) -> dict[str | None, int]:
        """Customers per quadrant; None counts those not scored yet."""
        ...


class Busy(Exception):
    """The store gave up: too many events for one customer at the same moment.

    Nothing was saved, so sending the event again is safe.
    """


class Store(Reader, Browser, Protocol):
    def handle(self, event: dict[str, Any], work: Work) -> Handled:
        """Run work, then save what it changed, as one unit.

        No other event's change lands between work's reads and the save: the
        event, the customer, the order if any, the history entry if an order
        was scored, and the outbox entry if the store has an outbox are saved
        together, or not at all. Raises Busy if the store gives up.
        """
        ...

    def unsent(self, before: datetime, limit: int) -> list[OutboxEntry]:
        """Up to limit outbox entries created before `before`, oldest first."""
        ...

    def sent(self, event_id: str) -> None:
        """Delete event_id's outbox entry: Pub/Sub has every message in it."""
        ...

    def list_models(self) -> list[ModelEntry]:
        """Every registered version, newest first (T310)."""
        ...

    def add_model(self, entry: ModelEntry) -> None:
        """Register a version. Raises ValueError if it is already registered."""
        ...

    def promote_model(self, version: str, now: datetime) -> ModelEntry:
        """Make version live and retire the live one before it, as one unit;
        the live version stays as it is. Raises UnknownVersion."""
        ...


def promoted(
    entries: dict[str, ModelEntry], version: str, now: datetime
) -> list[ModelEntry]:
    """The entries promoting version changes: it, now live, and the version
    live before it, now retired. Nothing when it is live already."""
    if version not in entries:
        raise UnknownVersion(version)
    if entries[version].status == LIVE:
        return []
    changed = [
        replace(entry, status="retired")
        for entry in entries.values()
        if entry.status == LIVE
    ]
    return [*changed, replace(entries[version], status=LIVE, promoted_at=now)]


def newest_first(entries: list[ModelEntry]) -> list[ModelEntry]:
    return sorted(entries, key=lambda e: (e.created_at, e.version), reverse=True)


def _copied(record: EventRecord) -> EventRecord:
    return EventRecord(copy.deepcopy(record.event), copy.deepcopy(record.result))


def _as_firestore(value: Any) -> Any:
    """A copy of value as Firestore gives it back: timestamps in UTC."""
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, dict):
        return {key: _as_firestore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_firestore(item) for item in value]
    return value


def customer_key(document: dict[str, Any]) -> CustomerKey:
    return document["churn_risk"], document["customer_id"]


def _customer_rank(key: CustomerKey) -> tuple[bool, float, str]:
    """Sorts as Firestore does: a missing churn risk (null) below any number."""
    churn_risk, customer_id = key
    return churn_risk is not None, churn_risk or 0.0, customer_id


def score_key(document: dict[str, Any]) -> TimeKey:
    return document["at"], document["event_id"]


def order_key(document: dict[str, Any]) -> TimeKey:
    return document["placed_at"], document["order_id"]


def _page(
    documents: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
    rank: Callable[[Any], Any],
    limit: int,
    after: Any,
) -> Page:
    """The page of documents, ranked high to low, that starts after `after`."""
    ranked = sorted(documents, key=lambda document: rank(key(document)), reverse=True)
    if after is not None:
        ranked = [document for document in ranked if rank(key(document)) < rank(after)]
    page = [copy.deepcopy(document) for document in ranked[:limit]]
    return Page(page, key(page[-1]) if len(ranked) > limit else None)


class MemoryStore:
    """A Store in this process's memory; it is lost when the process stops.

    One lock runs one event at a time, as a transaction would.
    """

    def __init__(self, *, outbox: bool = False) -> None:
        self.events: dict[str, EventRecord] = {}
        self.customers: dict[str, Customer] = {}
        self.orders: dict[str, Order] = {}
        self.history: dict[str, list[HistoryEntry]] = {}
        self.has_outbox = outbox
        self.outbox: dict[str, OutboxEntry] = {}
        # Every document as Firestore would hold it, by path: what Browser reads.
        self.documents: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def find_event(self, event_id: str) -> EventRecord | None:
        record = self.events.get(event_id)
        return None if record is None else _copied(record)

    def find_customer(self, customer_id: str) -> Customer | None:
        return self.customers.get(customer_id)

    def find_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    def find_live_model(self) -> ModelEntry | None:
        # No lock, as the other finds: handle() calls work under the lock.
        live = [
            entry_from_document(document)
            for path, document in list(self.documents.items())
            if path.startswith("models/") and document["status"] == LIVE
        ]
        return live[0] if live else None

    def handle(self, event: dict[str, Any], work: Work) -> Handled:
        entry = None
        with self._lock:
            result, applied = work(self)
            if applied is not None:
                record = EventRecord(event, result)
                self.save(record, applied, result["model_version"])
                if self.has_outbox:
                    documents = writes(record, applied, result["model_version"])
                    now = datetime.now(UTC)
                    entry = outbox_entry(event["event_id"], documents, now)
                    outbox_document(entry)  # checks what Firestore would get
                    self.outbox[entry.event_id] = entry
        return Handled(result, applied, entry)

    def unsent(self, before: datetime, limit: int) -> list[OutboxEntry]:
        with self._lock:
            old = [e for e in self.outbox.values() if e.created_at < before]
        return sorted(old, key=lambda entry: entry.created_at)[:limit]

    def sent(self, event_id: str) -> None:
        with self._lock:
            self.outbox.pop(event_id, None)

    def save(self, record: EventRecord, applied: Applied, model_version: str) -> None:
        """Save one event's changes; handle() calls it under the lock."""
        for path, document in writes(record, applied, model_version):
            self.documents[path] = _as_firestore(document)
        customer = applied.customer
        self.events[record.event["event_id"]] = _copied(record)
        self.customers[customer.customer_id] = customer
        if applied.order is not None:
            self.orders[applied.order.order_id] = applied.order
        entry = history_entry(record, applied, model_version)
        if entry is not None:
            self.history.setdefault(customer.customer_id, []).append(entry)

    # --- Browser: the read endpoints (T311) ---

    def _found(
        self, keep: Callable[[str, dict[str, Any]], bool]
    ) -> list[dict[str, Any]]:
        with self._lock:
            return [doc for path, doc in self.documents.items() if keep(path, doc)]

    def customer_document(self, customer_id: str) -> dict[str, Any] | None:
        with self._lock:
            found = self.documents.get(f"customers/{customer_id}")
        return copy.deepcopy(found)

    def list_customers(
        self,
        *,
        quadrant: str | None,
        min_churn_risk: float | None,
        limit: int,
        after: CustomerKey | None,
    ) -> Page:
        def keep(path: str, doc: dict[str, Any]) -> bool:
            risk = doc.get("churn_risk")
            return (
                path.startswith("customers/")
                and path.count("/") == 1
                and (quadrant is None or doc["quadrant"] == quadrant)
                and (
                    min_churn_risk is None
                    or (risk is not None and risk >= min_churn_risk)
                )
            )

        return _page(self._found(keep), customer_key, _customer_rank, limit, after)

    def list_scores(
        self, customer_id: str, *, limit: int, after: TimeKey | None
    ) -> Page:
        prefix = f"customers/{customer_id}/scores/"
        found = self._found(lambda path, _: path.startswith(prefix))
        return _page(found, score_key, lambda key: key, limit, after)

    def list_orders(
        self, customer_id: str, *, limit: int, after: TimeKey | None
    ) -> Page:
        found = self._found(
            lambda path, doc: (
                path.startswith("orders/") and doc["customer_id"] == customer_id
            )
        )
        return _page(found, order_key, lambda key: key, limit, after)

    def count_quadrants(self) -> dict[str | None, int]:
        customers = self._found(
            lambda path, _: path.startswith("customers/") and path.count("/") == 1
        )
        counts: dict[str | None, int] = {quadrant: 0 for quadrant in (*QUADRANTS, None)}
        for doc in customers:
            counts[doc["quadrant"]] += 1
        return counts

    # --- The model registry (T310) ---

    def _models(self) -> list[ModelEntry]:
        found = self._found(lambda path, _: path.startswith("models/"))
        return [entry_from_document(document) for document in found]

    def list_models(self) -> list[ModelEntry]:
        return newest_first(self._models())

    def add_model(self, entry: ModelEntry) -> None:
        document = model_document(entry)
        with self._lock:
            if f"models/{entry.version}" in self.documents:
                raise ValueError(f"{entry.version} is already registered")
            self.documents[f"models/{entry.version}"] = _as_firestore(document)

    def promote_model(self, version: str, now: datetime) -> ModelEntry:
        with self._lock:
            entries = {
                doc["version"]: entry_from_document(doc)
                for path, doc in self.documents.items()
                if path.startswith("models/")
            }
            for entry in promoted(entries, version, now):
                self.documents[f"models/{entry.version}"] = _as_firestore(
                    model_document(entry)
                )
            return entry_from_document(self.documents[f"models/{version}"])
