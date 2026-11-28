"""The Store in Firestore: one transaction per event (T302).

The documents and their schemas are in fitcore.documents (T301), and
cfi_api.store.writes lists the ones an event saves, checked. handle() reads,
scores and writes one event inside a Firestore transaction, so an event for
a customer that another event changed meanwhile is scored again on top of
that change; nothing is overwritten. After MAX_ATTEMPTS tries it gives up
with Busy, and the API asks the sender to try again.

With outbox=True the transaction also writes the event's outbox entry
(cfi_api.export, T306). Its time is read after the transaction's reads, so
when two events change one document, the one that commits later has the
later time: that is the order BigQuery's views go by.
"""

from datetime import UTC, datetime
from typing import Any

from fitcore.documents import check, customer_from_document, order_from_document
from fitcore.model import QUADRANTS
from fitcore.scoring import Customer, Order
from google.api_core.exceptions import Aborted, AlreadyExists
from google.cloud import firestore
from google.cloud.firestore import FieldFilter
from google.cloud.firestore_v1.field_path import FieldPath

from cfi_api.export import (
    OutboxEntry,
    outbox_document,
    outbox_entry,
    outbox_from_document,
)
from cfi_api.model_registry import LIVE, ModelEntry, entry_from_document, model_document
from cfi_api.store import (
    Busy,
    CustomerKey,
    EventRecord,
    Handled,
    Overview,
    Page,
    TimeKey,
    Work,
    by_name,
    customer_key,
    newest_first,
    order_key,
    promoted,
    score_key,
    writes,
)

DESCENDING = firestore.Query.DESCENDING
# The document id, as order_by and start_after name it.
NAME = FieldPath.document_id()

# The composite indexes the read queries need: (collection, fields in
# order). firestore.indexes.json must list them, and CI deploys it; the
# emulator runs any query without one, so only this list and the test that
# compares it with the file stand in for real Firestore.
COMPOSITE_INDEXES = (
    ("customers", (("quadrant", "ASCENDING"), ("churn_risk", "DESCENDING"))),
    ("orders", (("customer_id", "ASCENDING"), ("placed_at", "DESCENDING"))),
)

# Score history is queried on this field alone (list_scores); every other
# field of a score entry is left out of Firestore's indexes in
# firestore.indexes.json (T317). Questions across scorings go to BigQuery.
SCORES_ORDER = "at"


def _average(value: Any) -> float | None:
    """An avg() result: None when no document had the field."""
    return None if value is None else float(value)


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

    def find_live_model(self) -> ModelEntry | None:
        # A query, inside the event's transaction when it has one: a promote
        # then changes what this read, and Firestore runs the event again.
        query = (
            self._client.collection("models")
            .where(filter=FieldFilter("status", "==", LIVE))
            .limit(2)
        )
        found = [
            snapshot.to_dict()
            for snapshot in (
                self._transaction.get(query)
                if self._transaction is not None
                else query.stream()
            )
        ]
        if len(found) > 1:
            raise RuntimeError("more than one model version is live")
        return entry_from_document(found[0]) if found and found[0] else None


class FirestoreStore(FirestoreReader):
    """A Store in Firestore, through the client it is given."""

    def __init__(self, client: firestore.Client, *, outbox: bool = False) -> None:
        super().__init__(client)
        self._outbox = outbox

    def handle(self, event: dict[str, Any], work: Work) -> Handled:
        @firestore.transactional
        def run(transaction: firestore.Transaction) -> Handled:
            result, applied = work(FirestoreReader(self._client, transaction))
            if applied is None:
                return Handled(result, None, None)
            record, now = EventRecord(event, result), datetime.now(UTC)
            documents = writes(record, applied, result["model_version"], now)
            entry = None
            if self._outbox:
                entry = outbox_entry(event["event_id"], documents, now)
                documents.append((f"outbox/{entry.event_id}", outbox_document(entry)))
            for path, document in documents:
                transaction.set(self._client.document(path), document)
            return Handled(result, applied, entry)

        try:
            done: Handled = run(self._client.transaction(max_attempts=MAX_ATTEMPTS))
        except (Aborted, ValueError) as err:
            if gave_up(err):
                raise Busy(str(err)) from err
            raise
        return done

    def unsent(self, before: datetime, limit: int) -> list[OutboxEntry]:
        query = (
            self._client.collection("outbox")
            .where(filter=FieldFilter("created_at", "<", before))
            .order_by("created_at")
            .limit(limit)
        )
        found = (snapshot.to_dict() for snapshot in query.stream())
        return [outbox_from_document(document) for document in found if document]

    def sent(self, event_id: str) -> None:
        self._client.document(f"outbox/{event_id}").delete()

    # --- Browser: the read endpoints (T311) ---
    # Each list is ordered high to low, ending with the document id, so a
    # page resumes exactly after the last one. Lists that filter on one field
    # and sort on another need the composite indexes in firestore.indexes.json.

    def customer_document(self, customer_id: str) -> dict[str, Any] | None:
        return self._get(f"customers/{customer_id}")

    def list_customers(
        self,
        *,
        quadrant: str | None,
        min_churn_risk: float | None,
        limit: int,
        after: CustomerKey | None,
    ) -> Page:
        query: Any = self._client.collection("customers")
        if quadrant is not None:
            query = query.where(filter=FieldFilter("quadrant", "==", quadrant))
        if min_churn_risk is not None:
            query = query.where(filter=FieldFilter("churn_risk", ">=", min_churn_risk))
        query = query.order_by("churn_risk", direction=DESCENDING).order_by(
            NAME, direction=DESCENDING
        )
        if after is not None:
            query = query.start_after({"churn_risk": after[0], NAME: after[1]})
        return _page(query, customer_key, limit)

    def list_scores(
        self, customer_id: str, *, limit: int, after: TimeKey | None
    ) -> Page:
        query: Any = (
            self._client.collection(f"customers/{customer_id}/scores")
            .order_by(SCORES_ORDER, direction=DESCENDING)
            .order_by(NAME, direction=DESCENDING)
        )
        if after is not None:
            query = query.start_after({SCORES_ORDER: after[0], NAME: after[1]})
        return _page(query, score_key, limit)

    def list_orders(
        self, customer_id: str, *, limit: int, after: TimeKey | None
    ) -> Page:
        query: Any = (
            self._client.collection("orders")
            .where(filter=FieldFilter("customer_id", "==", customer_id))
            .order_by("placed_at", direction=DESCENDING)
            .order_by(NAME, direction=DESCENDING)
        )
        if after is not None:
            query = query.start_after({"placed_at": after[0], NAME: after[1]})
        return _page(query, order_key, limit)

    def count_quadrants(self) -> dict[str | None, int]:
        # Typed loosely: the client's hints say count().get() gives a flat
        # list, but it gives one list of results per answer.
        customers: Any = self._client.collection("customers")
        counts: dict[str | None, int] = {}
        for quadrant in (*QUADRANTS, None):
            query = customers.where(filter=FieldFilter("quadrant", "==", quadrant))
            # One aggregation per quadrant: billed by index entries, not documents.
            counts[quadrant] = int(query.count().get()[0][0].value)
        return counts

    def overview(self, at_risk: float) -> Overview:
        customers: Any = self._client.collection("customers")
        # Aggregations, billed by index entries read (one read per 1,000);
        # avg() skips customers not scored yet (no scores, churn_risk null).
        # One field per aggregation: two would need a composite index.
        risky = customers.where(filter=FieldFilter("churn_risk", ">=", at_risk))
        return Overview(
            counts=self.count_quadrants(),
            average_fit=_average(customers.avg("scores.fit").get()[0][0].value),
            average_churn_risk=_average(customers.avg("churn_risk").get()[0][0].value),
            at_risk=int(risky.count().get()[0][0].value),
        )

    # --- The model registry (T310) ---

    def list_models(self) -> list[ModelEntry]:
        found = (
            snapshot.to_dict()
            for snapshot in self._client.collection("models").stream()
        )
        return newest_first([entry_from_document(doc) for doc in found if doc])

    def add_model(self, entry: ModelEntry) -> None:
        try:
            self._client.document(f"models/{entry.version}").create(
                model_document(entry)
            )
        except AlreadyExists as err:
            raise ValueError(f"{entry.version} is already registered") from err

    def promote_model(self, version: str, now: datetime) -> ModelEntry:
        @firestore.transactional
        def run(transaction: firestore.Transaction) -> None:
            every = self._client.collection("models").order_by("created_at")
            snapshots = transaction.get(every)
            entries = {
                doc["version"]: entry_from_document(doc)
                for doc in (snapshot.to_dict() for snapshot in snapshots)
                if doc
            }
            for entry in promoted(entries, version, now):
                transaction.set(
                    self._client.document(f"models/{entry.version}"),
                    model_document(entry),
                )

        run(self._client.transaction(max_attempts=MAX_ATTEMPTS))
        found = self._client.document(f"models/{version}").get().to_dict()
        assert found is not None
        return entry_from_document(found)

    # --- Automation rules (T406) ---

    def list_rules(self) -> list[dict[str, Any]]:
        found = (
            snapshot.to_dict() for snapshot in self._client.collection("rules").stream()
        )
        return by_name([document for document in found if document])

    def add_rule(self, document: dict[str, Any]) -> None:
        check("rules", document)
        try:
            self._client.document(f"rules/{document['rule_id']}").create(document)
        except AlreadyExists as err:
            raise ValueError(f"{document['rule_id']} is there already") from err


def gave_up(err: Exception) -> bool:
    """Whether err is Firestore giving up on contention: Aborted (a read gave
    up waiting for another event), or the client's ValueError once every
    commit was aborted. Any other ValueError is a real error."""
    if isinstance(err, Aborted):
        return True
    return isinstance(err, ValueError) and isinstance(err.__cause__, Aborted)


def _page(query: Any, key: Any, limit: int) -> Page:
    """The first limit documents query finds, and whether there are more."""
    found = [snapshot.to_dict() for snapshot in query.limit(limit + 1).stream()]
    page = [document for document in found[:limit] if document is not None]
    return Page(page, key(page[-1]) if len(found) > limit else None)
