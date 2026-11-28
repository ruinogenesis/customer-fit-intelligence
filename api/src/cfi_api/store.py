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

So do the automation rules (T406): a Browser lists them, by name, for
GET /rules, and a Store adds one (python -m cfi_api.rules seed) and sets
its status. The rules engine (T501) logs each decision as an action, once,
reads the guardrails' settings and a customer's recent actions, and counts
each day's spend in budget/{day} as it logs a decision (T502).
"""

import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
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
    record: EventRecord,
    applied: Applied,
    model_version: str,
    received_at: datetime,
) -> list[tuple[str, dict[str, Any]]]:
    """The documents one event saves, as (path, document), each checked.

    The event's document expires EVENT_RETENTION after received_at, the
    API's clock: never the event's own occurred_at, which a client or a
    simulation sets.

    Raises fitcore.documents.DocumentError if one does not fit its schema.
    """
    customer_id = applied.customer.customer_id
    event = {
        "event": record.event,
        "result": record.result,
        EXPIRES_AT: received_at + EVENT_RETENTION,
    }
    documents = [
        ("events", f"events/{record.event['event_id']}", event),
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


EVENT_RETENTION = timedelta(days=30)
"""How long Firestore keeps events/{event_id} after the API received it.

Firestore's TTL policy on EXPIRES_AT (firestore.indexes.json) deletes the
document, usually within a day after it expires; BigQuery keeps every event.
The same event_id is recognised as a repeat only while its document is
there: sent again later, it counts as new (an order_placed is still refused,
since orders/{order_id} stays).
"""

EXPIRES_AT = "expires_at"
"""The TTL field on events/{event_id}: a timestamp, never indexed."""


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

    def overview(self, at_risk: float) -> "Overview":
        """Every customer by quadrant, their averages, and the at-risk count."""
        ...

    def list_rules(self) -> list[dict[str, Any]]:
        """Every automation rule, rules/{rule_id}, by name (T406)."""
        ...


class Overview(NamedTuple):
    """The Overview screen's numbers now (T402). The averages are over the
    customers scored so far: None while there are none."""

    counts: dict[str | None, int]
    average_fit: float | None
    average_churn_risk: float | None
    at_risk: int


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

    def add_rule(self, document: dict[str, Any]) -> None:
        """Add rules/{rule_id}, checked first. Raises ValueError if that rule
        is there already (T406)."""
        ...

    def set_rule_status(
        self, rule_id: str, status: str, now: datetime
    ) -> dict[str, Any]:
        """Set a rule's status and updated_at; the rule as it is now. Raises
        UnknownRule, or ValueError for a status the schema refuses (T501)."""
        ...

    def add_action(
        self,
        document: dict[str, Any],
        *,
        day: str | None = None,
        budget: int | None = None,
    ) -> bool:
        """Add actions/{action_id}, checked first; False if it is there
        already, unchanged (T501).

        With a day and a budget (T502): raise OverBudget, adding nothing, if
        the action's cost would take budget/{day} past budget, whichever its
        arm; a treatment action's cost is added to that day's spend in the
        same unit as the action itself.
        """
        ...

    def customer_actions(
        self, customer_id: str, since: datetime
    ) -> list[dict[str, Any]]:
        """A customer's actions decided at since or later (T502)."""
        ...

    def quiet_customers(self, before: datetime) -> list[dict[str, Any]]:
        """The customers whose latest order was placed at before or earlier,
        as customers/{customer_id}; none who never ordered (T505)."""
        ...

    def automation_settings(self) -> dict[str, Any] | None:
        """settings/automations, or None if it was never set (T502)."""
        ...

    def set_automation_settings(self, document: dict[str, Any]) -> None:
        """Write settings/automations, checked first (T502)."""
        ...

    def spent_on(self, day: str) -> int:
        """What the treatment arm spent on day, in satang (T502)."""
        ...

    def list_actions(self, limit: int) -> list[dict[str, Any]]:
        """The latest limit actions, newest first (T501)."""
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


def by_name(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rules, key=lambda rule: (rule["name"].casefold(), rule["rule_id"]))


class UnknownRule(KeyError):
    """No rule has this rule_id."""


class OverBudget(Exception):
    """An action's cost would take the day past the daily budget (T502)."""


TREATMENT = "treatment"  # the arm that acts, so the one that spends


def spend(
    action: dict[str, Any], day: str, spent: int, budget: int
) -> dict[str, Any] | None:
    """The budget/{day} document after logging action, given what the day had
    spent: None if the action spends nothing. Raises OverBudget."""
    worth = action["cost"]
    if worth == 0:
        return None
    if spent + worth > budget:
        raise OverBudget(f"{day}: {spent} + {worth} > {budget} satang")
    if action["arm"] != TREATMENT:
        return None
    document = {"day": day, "spent": spent + worth, "updated_at": action["decided_at"]}
    check("budget", document)
    return document


def with_status(rule: dict[str, Any], status: str, now: datetime) -> dict[str, Any]:
    """rule with a new status and updated_at, checked."""
    changed = {**rule, "status": status, "updated_at": now}
    check("rules", changed)
    return changed


def newest_actions(actions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(actions, key=lambda a: (a["decided_at"], a["action_id"]))
    return ranked[::-1][:limit]


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
                version, now = result["model_version"], datetime.now(UTC)
                self.save(record, applied, version, now)
                if self.has_outbox:
                    documents = writes(record, applied, version, now)
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

    def save(
        self,
        record: EventRecord,
        applied: Applied,
        model_version: str,
        received_at: datetime,
    ) -> None:
        """Save one event's changes; handle() calls it under the lock."""
        for path, document in writes(record, applied, model_version, received_at):
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

    def overview(self, at_risk: float) -> Overview:
        customers = self._found(
            lambda path, _: path.startswith("customers/") and path.count("/") == 1
        )
        scored = [doc for doc in customers if doc["scores"] is not None]
        fits = [doc["scores"]["fit"] for doc in scored]
        risks = [doc["churn_risk"] for doc in scored]
        return Overview(
            counts=self.count_quadrants(),
            average_fit=sum(fits) / len(fits) if fits else None,
            average_churn_risk=sum(risks) / len(risks) if risks else None,
            at_risk=sum(risk >= at_risk for risk in risks),
        )

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

    # --- Automation rules (T406) ---

    def list_rules(self) -> list[dict[str, Any]]:
        found = self._found(lambda path, _: path.startswith("rules/"))
        return by_name(copy.deepcopy(found))

    def add_rule(self, document: dict[str, Any]) -> None:
        check("rules", document)
        path = f"rules/{document['rule_id']}"
        with self._lock:
            if path in self.documents:
                raise ValueError(f"{document['rule_id']} is there already")
            self.documents[path] = _as_firestore(document)

    def set_rule_status(
        self, rule_id: str, status: str, now: datetime
    ) -> dict[str, Any]:
        path = f"rules/{rule_id}"
        with self._lock:
            if path not in self.documents:
                raise UnknownRule(rule_id)
            changed = with_status(self.documents[path], status, now)
            self.documents[path] = _as_firestore(changed)
            return copy.deepcopy(self.documents[path])

    # --- The rules engine's decisions (T501) and guardrails (T502) ---

    def add_action(
        self,
        document: dict[str, Any],
        *,
        day: str | None = None,
        budget: int | None = None,
    ) -> bool:
        check("actions", document)
        path = f"actions/{document['action_id']}"
        with self._lock:
            if path in self.documents:
                return False
            if day is not None and budget is not None:
                spent = self.documents.get(f"budget/{day}", {}).get("spent", 0)
                day_after = spend(document, day, spent, budget)
                if day_after is not None:
                    self.documents[f"budget/{day}"] = _as_firestore(day_after)
            self.documents[path] = _as_firestore(document)
            return True

    def customer_actions(
        self, customer_id: str, since: datetime
    ) -> list[dict[str, Any]]:
        found = self._found(
            lambda path, doc: (
                path.startswith("actions/")
                and doc["customer_id"] == customer_id
                and doc["decided_at"] >= since
            )
        )
        return copy.deepcopy(found)

    def quiet_customers(self, before: datetime) -> list[dict[str, Any]]:
        found = self._found(
            lambda path, doc: (
                path.count("/") == 1
                and path.startswith("customers/")
                and doc.get("latest_placed_at") is not None
                and doc["latest_placed_at"] <= before
            )
        )
        return copy.deepcopy(found)

    def automation_settings(self) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self.documents.get("settings/automations"))

    def set_automation_settings(self, document: dict[str, Any]) -> None:
        check("settings", document)
        with self._lock:
            self.documents["settings/automations"] = _as_firestore(document)

    def spent_on(self, day: str) -> int:
        with self._lock:
            found = self.documents.get(f"budget/{day}")
        return 0 if found is None else int(found["spent"])

    def list_actions(self, limit: int) -> list[dict[str, Any]]:
        found = self._found(lambda path, _: path.startswith("actions/"))
        return newest_actions(copy.deepcopy(found), limit)
