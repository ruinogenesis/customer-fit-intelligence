"""The Store in Firestore: one transaction per event (T302).

The documents and their schemas are in fitcore.documents (T301), and
cfi_api.store.writes lists the ones an event saves, checked. handle() reads,
scores and writes one event inside a Firestore transaction, so an event for
a customer that another event changed meanwhile is scored again on top of
that change; nothing is overwritten. After MAX_ATTEMPTS tries it gives up
with Busy, and the API asks the sender to try again.
"""

from typing import Any

from fitcore.documents import customer_from_document, order_from_document
from fitcore.scoring import Applied, Customer, Order
from google.api_core.exceptions import Aborted
from google.cloud import firestore

from cfi_api.store import Busy, EventRecord, Work, writes

# Tries per event before giving up: the client library's default is 5.
MAX_ATTEMPTS = 10


class FirestoreReader:
    """A Reader of Firestore, inside a transaction if it is given one."""

    def __init__(
        self,
        client: firestore.Client,
        transaction: firestore.Transaction | None = None,
    ) -> None:
        self._client = client
        self._transaction = transaction

    def _get(self, path: str) -> dict[str, Any] | None:
        snapshot = self._client.document(path).get(transaction=self._transaction)
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


class FirestoreStore(FirestoreReader):
    """A Store in Firestore, through the client it is given."""

    def __init__(self, client: firestore.Client) -> None:
        super().__init__(client)

    def handle(
        self, event: dict[str, Any], work: Work
    ) -> tuple[dict[str, Any], Applied | None]:
        @firestore.transactional
        def run(
            transaction: firestore.Transaction,
        ) -> tuple[dict[str, Any], Applied | None]:
            result, applied = work(FirestoreReader(self._client, transaction))
            if applied is not None:
                record = EventRecord(event, result)
                for path, document in writes(record, applied, result["model_version"]):
                    transaction.set(self._client.document(path), document)
            return result, applied

        try:
            done: tuple[dict[str, Any], Applied | None] = run(
                self._client.transaction(max_attempts=MAX_ATTEMPTS)
            )
        except (Aborted, ValueError) as err:
            if gave_up(err):
                raise Busy(str(err)) from err
            raise
        return done


def gave_up(err: Exception) -> bool:
    """Whether err is Firestore giving up on contention: Aborted (a read gave
    up waiting for another event), or the client's ValueError once every
    commit was aborted. Any other ValueError is a real error."""
    if isinstance(err, Aborted):
        return True
    return isinstance(err, ValueError) and isinstance(err.__cause__, Aborted)
