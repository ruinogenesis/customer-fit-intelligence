"""Where the API keeps events, customers, orders and score history.

Store is what the events endpoint needs, and Reader is its reading half:
all that what-if scoring gets. MemoryStore keeps everything in this process
(tests, and trying the API without Firestore);
cfi_api.firestore_store.FirestoreStore keeps it in Firestore. One event is
saved with one call, so a store can write it all at once.
"""

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from fitcore.scoring import Applied, Customer, Order, Scores


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
    model_version: str


def history_entry(
    record: EventRecord, applied: Applied, model_version: str
) -> HistoryEntry | None:
    """The history entry for an event that scored an order, else None."""
    if applied.scored is None or applied.order is None:
        return None
    return HistoryEntry(
        event_id=record.event["event_id"],
        order_id=applied.order.order_id,
        at=record.event["occurred_at"],
        scores=applied.scored,
        model_version=model_version,
    )


class Reader(Protocol):
    """The reading half of a Store.

    Code given a Reader cannot save: mypy refuses the call. That is how
    POST /score stays write-free.
    """

    def find_event(self, event_id: str) -> EventRecord | None: ...

    def find_customer(self, customer_id: str) -> Customer | None: ...

    def find_order(self, order_id: str) -> Order | None: ...


class Store(Reader, Protocol):
    def save(self, record: EventRecord, applied: Applied, model_version: str) -> None:
        """Save the event, the customer, the order if any, and the history
        entry if an order was scored: all of them, or none."""
        ...


def _copied(record: EventRecord) -> EventRecord:
    return EventRecord(copy.deepcopy(record.event), copy.deepcopy(record.result))


class MemoryStore:
    """A Store in this process's memory; it is lost when the process stops."""

    def __init__(self) -> None:
        self.events: dict[str, EventRecord] = {}
        self.customers: dict[str, Customer] = {}
        self.orders: dict[str, Order] = {}
        self.history: dict[str, list[HistoryEntry]] = {}

    def find_event(self, event_id: str) -> EventRecord | None:
        record = self.events.get(event_id)
        return None if record is None else _copied(record)

    def find_customer(self, customer_id: str) -> Customer | None:
        return self.customers.get(customer_id)

    def find_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    def save(self, record: EventRecord, applied: Applied, model_version: str) -> None:
        customer = applied.customer
        self.events[record.event["event_id"]] = _copied(record)
        self.customers[customer.customer_id] = customer
        if applied.order is not None:
            self.orders[applied.order.order_id] = applied.order
        entry = history_entry(record, applied, model_version)
        if entry is not None:
            self.history.setdefault(customer.customer_id, []).append(entry)
