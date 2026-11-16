"""The Store in Firestore: one batch of writes per event.

The documents and their schemas are in fitcore.documents (T301), and
cfi_api.store.writes lists the ones an event saves, checked. An event's
writes land together or not at all. The reads before them are not in a
transaction yet, so two events for one customer at the same moment can still
overwrite each other's changes (T302).
"""

from typing import Any

from fitcore.documents import customer_from_document, order_from_document
from fitcore.scoring import Applied, Customer, Order
from google.cloud import firestore

from cfi_api.store import EventRecord, writes


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
        batch = self._client.batch()
        for path, document in writes(record, applied, model_version):
            batch.set(self._client.document(path), document)
        batch.commit()
