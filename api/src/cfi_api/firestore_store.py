"""The Store in Firestore: one batch of writes per event.

Documents (T301 will document and validate them):

- events/{event_id}: the event as received, and the result it got.
- customers/{customer_id}: expectations (x2..x8), latest order, scores,
  churn_risk, quadrant and the model_version that wrote them.
- customers/{customer_id}/scores/{event_id}: one entry per scoring.
- orders/{order_id}: customer, placed_at, promised_eta and the gaps so far.

An event's writes land together or not at all. The reads before them are
not in a transaction yet, so two events for one customer at the same moment
can still overwrite each other's changes (T302).
"""

from collections.abc import Mapping
from typing import Any

from fitcore.expectations import Baseline
from fitcore.model import LATENTS
from fitcore.scoring import Applied, Customer, Order, Scores
from google.cloud import firestore

from cfi_api.store import EventRecord, HistoryEntry, history_entry


def _scores(scores: Scores) -> dict[str, float]:
    return {name: getattr(scores, name) for name in LATENTS}


def customer_document(customer: Customer, model_version: str) -> dict[str, Any]:
    scores = customer.scores
    return {
        "customer_id": customer.customer_id,
        "expectations": {
            key: {"mean": b.mean, "weight": b.weight, "last_seen": b.last_seen}
            for key, b in customer.baselines.items()
        },
        "latest_order_id": customer.latest_order_id,
        "latest_placed_at": customer.latest_placed_at,
        "scores": None if scores is None else _scores(scores),
        "churn_risk": None if scores is None else scores.churn_risk,
        "quadrant": None if scores is None else scores.quadrant,
        "model_version": model_version,
    }


def customer_from_document(document: Mapping[str, Any]) -> Customer:
    scores = None
    if document["scores"] is not None:
        scores = Scores(
            **document["scores"],
            churn_risk=document["churn_risk"],
            quadrant=document["quadrant"],
        )
    return Customer(
        customer_id=document["customer_id"],
        baselines={
            key: Baseline(**value) for key, value in document["expectations"].items()
        },
        latest_order_id=document["latest_order_id"],
        latest_placed_at=document["latest_placed_at"],
        scores=scores,
    )


def order_document(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "placed_at": order.placed_at,
        "promised_eta": order.promised_eta,
        "gaps": dict(order.gaps),
    }


def order_from_document(document: Mapping[str, Any]) -> Order:
    return Order(
        order_id=document["order_id"],
        customer_id=document["customer_id"],
        placed_at=document["placed_at"],
        promised_eta=document["promised_eta"],
        gaps=document["gaps"],
    )


def history_document(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "event_id": entry.event_id,
        "order_id": entry.order_id,
        "at": entry.at,
        "scores": _scores(entry.scores),
        "churn_risk": entry.scores.churn_risk,
        "quadrant": entry.scores.quadrant,
        "model_version": entry.model_version,
    }


class FirestoreStore:
    """A Store in Firestore, through the client it is given."""

    def __init__(self, client: firestore.Client) -> None:
        self._client = client

    def _get(self, path: str) -> dict[str, Any] | None:
        snapshot = self._client.document(path).get()
        return snapshot.to_dict() if snapshot.exists else None

    def find_event(self, event_id: str) -> EventRecord | None:
        document = self._get(f"events/{event_id}")
        if document is None:
            return None
        return EventRecord(event=document["event"], result=document["result"])

    def find_customer(self, customer_id: str) -> Customer | None:
        document = self._get(f"customers/{customer_id}")
        return None if document is None else customer_from_document(document)

    def find_order(self, order_id: str) -> Order | None:
        document = self._get(f"orders/{order_id}")
        return None if document is None else order_from_document(document)

    def save(self, record: EventRecord, applied: Applied, model_version: str) -> None:
        customer_id = applied.customer.customer_id
        batch = self._client.batch()
        event = {"event": record.event, "result": record.result}
        batch.set(self._client.document(f"events/{record.event['event_id']}"), event)
        batch.set(
            self._client.document(f"customers/{customer_id}"),
            customer_document(applied.customer, model_version),
        )
        if applied.order is not None:
            path = f"orders/{applied.order.order_id}"
            batch.set(self._client.document(path), order_document(applied.order))
        entry = history_entry(record, applied, model_version)
        if entry is not None:
            path = f"customers/{customer_id}/scores/{entry.event_id}"
            batch.set(self._client.document(path), history_document(entry))
        batch.commit()
