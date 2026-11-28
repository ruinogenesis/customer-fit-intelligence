"""The rules engine (T501): which rules fire on a new scoring, and their decisions.

The rules service (cfi_api.rules_service) hands decide() every new score
entry. Each rule that is not off and has its logic here (LOGIC) is asked
whether it fires; the numbers it fires on come from its condition map, over
the logic's own defaults. Each rule that fires is logged to
actions/{action_id} with the customer's arm for it (T503, folded into T501):
a stable hash of rule and customer puts the rule's holdout_share of
customers in the holdout, where the decision is logged and nothing is done.

Before a decision is logged, the guardrails (T502) check it, in this order;
the settings they share live in settings/automations (Guardrails, with
defaults when it is missing):

1. The global kill switch: with enabled false, no rule decides anything.
2. The rule's own kill switch: status off.
3. Consent (T610): a marketing rule decides only for a customer who holds
   marketing consent now. Service rules need none.
4. The frequency cap: at most cap_actions marketing decisions per customer
   over the last cap_days days, both arms counted, so the arms stay alike.
   Service rules are exempt.
5. The daily budget: each decision costs what it would spend (a voucher's
   value; 0 for a message). Only the treatment arm spends, but once a cost
   would take the local day past daily_budget, the rule decides for neither
   arm until the next day. The store counts the spend in the decision's own
   transaction, so two decisions at once cannot overspend.

A decision a guardrail stops logs nothing, only a "held back" line. Quiet
hours do not stop a decision: a marketing decision made during them gets a
send_at at their end, and the channels (T504) send nothing before it.
Service decisions go at once.

Until the channels (T504) exist, every decision is logged as a dry run,
whatever the rule's status: nothing is ever sent.

A rule decides at most once per order. An order is scored again as it is
delivered and rated, and Pub/Sub may send one scoring twice; action_id is
the rule and the order, so neither logs a second decision.

Some rules fire on what has not happened, which no scoring brings: the
daily sweep (T505) asks those (SWEEPS) which customers they fire for, and
puts each through the same guardrails. Win-back fires for a customer with
no order for 14 days; its decision is about their latest order, so it
decides once per quiet spell however many sweeps see them.
"""

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from fitcore.documents import check

from cfi_api.store import OverBudget

logger = logging.getLogger(__name__)

OFF = "off"
MARKETING = "marketing"
TREATMENT, HOLDOUT = "treatment", "holdout"


@dataclass(frozen=True)
class Scoring:
    """One new score entry: customers/{customer_id}/scores/{event_id}."""

    customer_id: str
    event_id: str
    entry: Mapping[str, Any]  # the score entry, as saved (fitcore.documents)


@dataclass(frozen=True)
class Fire:
    """A rule's decision to act, and the channel it would act through."""

    channel: str


# A rule's logic: the scoring and the rule's numbers in, a Fire or None out.
Logic = Callable[[Scoring, Mapping[str, Any]], Fire | None]


def numbers(given: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """A rule's numbers: its condition or action map over the logic's defaults.

    Raises ValueError for a name the logic does not know, so a typo is
    refused instead of ignored.
    """
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {sorted(unknown)}; known {sorted(defaults)}")
    return {**defaults, **given}


FEE_SENSITIVITY = {"quadrant": "product", "min_fee_drag": 0.05}


def fee_sensitivity(scoring: Scoring, condition: Mapping[str, Any]) -> Fire | None:
    """In the Product quadrant, with fees pulling fit down: a fee-free window.

    Fires when the order's quadrant is the condition's (product) and the fee
    load (x3) took at least min_fee_drag off fit, against an order exactly
    as expected (the scoring's exact split, T115). With the T112 model, fees
    one percentage point above what the customer expects take about 0.058.
    """
    rule = numbers(condition, FEE_SENSITIVITY)
    entry = scoring.entry
    drag = -entry["contributions"]["fit"]["x3"]
    if entry["quadrant"] == rule["quadrant"] and drag >= rule["min_fee_drag"]:
        return Fire("voucher")
    return None


# Each rule with logic, by rule_id; T506 adds the other starter rules.
LOGIC: dict[str, Logic] = {"fee-sensitivity": fee_sensitivity}

# What each rule's action is worth, in satang, before its action map: a
# fee-free delivery is worth about a typical delivery fee, and win-back's
# voucher 50 baht. A rule not here sends messages, which cost nothing.
ACTIONS: dict[str, dict[str, int]] = {
    "fee-sensitivity": {"value": 4_000},
    "win-back": {"value": 5_000},
}


def cost(rule: Mapping[str, Any]) -> int:
    """What a decision of rule would spend: its action map's value over the
    default. Raises ValueError for an unknown name or a value that is not a
    whole number of satang, 0 or more."""
    value = numbers(rule["action"], ACTIONS.get(rule["rule_id"], {"value": 0}))
    worth = value["value"]
    if not isinstance(worth, int) or isinstance(worth, bool) or worth < 0:
        raise ValueError(f"value must be whole satang, 0 or more; got {worth!r}")
    return worth


# --- The guardrails' settings (T502) ---


@dataclass(frozen=True)
class Guardrails:
    """settings/automations: the guardrails every rule shares."""

    enabled: bool = True
    cap_actions: int = 2
    cap_days: int = 7
    quiet_start: str = "21:00"
    quiet_end: str = "08:00"
    time_zone: str = "Asia/Bangkok"
    daily_budget: int = 500_000  # satang: 5,000 baht

    @classmethod
    def from_document(cls, document: Mapping[str, Any] | None) -> "Guardrails":
        """The settings in document, or the defaults if there is none."""
        if document is None:
            return cls()
        cap, quiet = document["frequency_cap"], document["quiet_hours"]
        return cls(
            enabled=document["enabled"],
            cap_actions=cap["actions"],
            cap_days=cap["days"],
            quiet_start=quiet["start"],
            quiet_end=quiet["end"],
            time_zone=quiet["time_zone"],
            daily_budget=document["daily_budget"],
        )

    def document(self, now: datetime) -> dict[str, Any]:
        """These settings as settings/automations, checked."""
        document = {
            "enabled": self.enabled,
            "frequency_cap": {"actions": self.cap_actions, "days": self.cap_days},
            "quiet_hours": {
                "start": self.quiet_start,
                "end": self.quiet_end,
                "time_zone": self.time_zone,
            },
            "daily_budget": self.daily_budget,
            "updated_at": now,
        }
        check("settings", document)
        return document

    def local_day(self, now: datetime) -> str:
        """now's day where the customers are, YYYY-MM-DD: the budget's day."""
        return now.astimezone(ZoneInfo(self.time_zone)).date().isoformat()

    def quiet_until(self, now: datetime) -> datetime | None:
        """When the quiet hours now falls in end, or None outside them."""
        zone = ZoneInfo(self.time_zone)
        local = now.astimezone(zone)
        start, end, clock = (
            _clock(self.quiet_start),
            _clock(self.quiet_end),
            local.time(),
        )
        if start == end:
            return None  # no quiet hours
        inside = start <= clock < end if start < end else clock >= start or clock < end
        if not inside:
            return None
        day = local.date() if clock < end else local.date() + timedelta(days=1)
        return datetime.combine(day, end, tzinfo=zone).astimezone(UTC)


def _clock(text: str) -> time:
    hours, minutes = text.split(":")
    return time(int(hours), int(minutes))


def arm(rule_id: str, customer_id: str, holdout_share: float) -> str:
    """The customer's arm for the rule: always the same for the same pair.

    The pair hashes to a point in [0, 1); below holdout_share is the holdout.
    So lowering the share, as a rule going from test (0.5) to live (0.1) does,
    only moves holdout customers to treatment, never the other way.
    """
    digest = hashlib.sha256(f"{rule_id}\x00{customer_id}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    return HOLDOUT if point < holdout_share else TREATMENT


def action_id(rule_id: str, order_id: str) -> str:
    return f"{rule_id}--{order_id}"


def action_document(
    rule: Mapping[str, Any],
    customer_id: str,
    order_id: str,
    fire: Fire,
    worth: int,
    now: datetime,
    send_at: datetime,
) -> dict[str, Any]:
    """The document for actions/{action_id}, checked; always a dry run until
    the channels (T504). order_id is the order the decision is about: the
    scored order, or the latest order before a customer went quiet."""
    document = {
        "action_id": action_id(rule["rule_id"], order_id),
        "rule_id": rule["rule_id"],
        "customer_id": customer_id,
        "arm": arm(rule["rule_id"], customer_id, rule["holdout_share"]),
        "channel": fire.channel,
        "dry_run": True,
        "decided_at": now,
        "cost": worth,
        "send_at": send_at,
        "outcome": None,
    }
    check("actions", document)
    return document


def consents(customer: Mapping[str, Any] | None) -> bool:
    """Whether a customer document holds marketing consent now (T610). A
    customer never asked has none: no field (saved before T610), or null."""
    consent = None if customer is None else customer.get("consent")
    return consent is not None and consent["marketing"] is True


class Automations(Protocol):
    """What the engine needs from a store."""

    def automation_settings(self) -> dict[str, Any] | None: ...

    def list_rules(self) -> list[dict[str, Any]]: ...

    def customer_document(self, customer_id: str) -> dict[str, Any] | None: ...

    def customer_actions(
        self, customer_id: str, since: datetime
    ) -> list[dict[str, Any]]: ...

    def quiet_customers(self, before: datetime) -> list[dict[str, Any]]: ...

    def add_action(self, document: dict[str, Any], *, day: str, budget: int) -> bool:
        """Add actions/{action_id} and count its spend on day; False if it is
        there already. Raises OverBudget past budget (cfi_api.store)."""
        ...


# --- The daily sweep (T505) ---


@dataclass(frozen=True)
class Candidate:
    """A customer a sweeping rule fires for, and the order it is about."""

    customer: Mapping[str, Any]  # customers/{customer_id}, as saved
    order_id: str
    fire: Fire


# A sweeping rule's logic: it finds its own customers, from the store and
# the rule's numbers, as of now.
Sweep = Callable[[Automations, Mapping[str, Any], datetime], list[Candidate]]

WIN_BACK = {"quiet_days": 14, "min_churn": 0.10}


def win_back(
    store: Automations, condition: Mapping[str, Any], now: datetime
) -> list[Candidate]:
    """No order for quiet_days, and the latest order's churn risk at least
    min_churn: a voucher to come back, riskiest first, so the daily budget
    goes to them first.

    The order is the customer's latest, so the rule decides once per quiet
    spell: after a new order and another quiet_days, it can decide again.
    """
    rule = numbers(condition, WIN_BACK)
    before = now - timedelta(days=rule["quiet_days"])
    found = [
        customer
        for customer in store.quiet_customers(before)
        if customer["churn_risk"] is not None
        and customer["churn_risk"] >= rule["min_churn"]
    ]
    found.sort(key=lambda customer: (-customer["churn_risk"], customer["customer_id"]))
    return [
        Candidate(customer, customer["latest_order_id"], Fire("voucher"))
        for customer in found
    ]


# Each rule that sweeps, by rule_id; T506 adds Lovemark advocacy.
SWEEPS: dict[str, Sweep] = {"win-back": win_back}


@dataclass
class _Seen:
    """What one run has read about one customer, read once and only if a
    marketing rule fires for them."""

    consent: bool | None = None
    recent: int | None = None  # their marketing decisions within the cap
    known: set[str] = field(default_factory=set)  # those actions' ids


class _Run:
    """One decide() or sweep(): the guardrails' settings, the rules, the local
    day and quiet hours, and what it has read about each customer."""

    def __init__(self, store: Automations, now: datetime, guard: Guardrails) -> None:
        self.store, self.now, self.guard = store, now, guard
        self.rules = store.list_rules()
        self.marketing = {r["rule_id"] for r in self.rules if r["kind"] == MARKETING}
        self.day, self.quiet = guard.local_day(now), guard.quiet_until(now)
        self.seen: dict[str, _Seen] = {}

    @classmethod
    def start(cls, store: Automations, now: datetime, who: str) -> "_Run | None":
        """A run, or None when automations are paused (the global switch)."""
        guard = Guardrails.from_document(store.automation_settings())
        if not guard.enabled:
            logger.info("Automations are paused: nothing decided for %s", who)
            return None
        return cls(store, now, guard)

    def active(self, logic: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The rules that are not off (their own kill switch) and have logic."""
        return [
            rule
            for rule in self.rules
            if rule["status"] != OFF and rule["rule_id"] in logic
        ]

    def consider(
        self,
        rule: Mapping[str, Any],
        customer_id: str,
        order_id: str,
        fire: Fire,
        worth: int,
        customer: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Put a rule's decision to fire through the guardrails, and log it if
        it passes: the action logged, or None. customer, when the caller has
        read it already, saves reading it again for its consent."""
        send_at = self.now
        if rule["kind"] == MARKETING:
            seen = self.seen.setdefault(customer_id, _Seen())
            if seen.consent is None:
                document = customer or self.store.customer_document(customer_id)
                seen.consent = consents(document)
            if not seen.consent:
                _held(rule, customer_id, "no marketing consent")
                return None
            if seen.recent is None:
                since = self.now - timedelta(days=self.guard.cap_days)
                found = self.store.customer_actions(customer_id, since)
                seen.recent = sum(a["rule_id"] in self.marketing for a in found)
                seen.known = {a["action_id"] for a in found}
            if action_id(rule["rule_id"], order_id) in seen.known:
                return None  # decided on this order already: nothing to hold back
            if seen.recent >= self.guard.cap_actions:
                cap = f"{self.guard.cap_actions} in {self.guard.cap_days} days"
                _held(rule, customer_id, f"frequency cap ({cap})")
                return None
            send_at = self.quiet or self.now
        document = action_document(
            rule, customer_id, order_id, fire, worth, self.now, send_at
        )
        try:
            added = self.store.add_action(
                document, day=self.day, budget=self.guard.daily_budget
            )
        except OverBudget:
            _held(rule, customer_id, f"daily budget ({self.day})")
            return None
        if not added:
            return None
        if rule["kind"] == MARKETING:
            seen = self.seen[customer_id]
            seen.recent = (seen.recent or 0) + 1
            seen.known.add(document["action_id"])
        return document


def _held(rule: Mapping[str, Any], customer_id: str, why: str) -> None:
    logger.info("%s held back for %s: %s", rule["rule_id"], customer_id, why)


def decide(store: Automations, scoring: Scoring, now: datetime) -> list[dict[str, Any]]:
    """Ask every rule that is on and has logic about scoring, and log each
    that fires and passes the guardrails. The decisions logged now, in rule
    order: none for a rule that had decided on this order already.

    A rule whose condition or action the logic refuses is skipped and logged
    as an error, so one bad rule never stops the others.
    """
    run = _Run.start(store, now, scoring.customer_id)
    if run is None:
        return []
    decided = []
    for rule in run.active(LOGIC):
        try:
            fire = LOGIC[rule["rule_id"]](scoring, rule["condition"])
            worth = cost(rule)
        except ValueError:
            logger.exception("Rule %s was skipped: its numbers", rule["rule_id"])
            continue
        if fire is None:
            continue
        action = run.consider(
            rule, scoring.customer_id, scoring.entry["order_id"], fire, worth
        )
        if action is not None:
            decided.append(action)
    return decided


def sweep(store: Automations, now: datetime) -> list[dict[str, Any]]:
    """The daily sweep (T505): ask every rule that is on and sweeps which
    customers it fires for, and log each decision that passes the
    guardrails, as decide() does. The decisions logged now, in rule order.
    """
    run = _Run.start(store, now, "the daily sweep")
    if run is None:
        return []
    decided = []
    for rule in run.active(SWEEPS):
        try:
            found = SWEEPS[rule["rule_id"]](store, rule["condition"], now)
            worth = cost(rule)
        except ValueError:
            logger.exception("Rule %s was skipped: its numbers", rule["rule_id"])
            continue
        for candidate in found:
            customer = candidate.customer
            action = run.consider(
                rule,
                customer["customer_id"],
                candidate.order_id,
                candidate.fire,
                worth,
                customer,
            )
            if action is not None:
                decided.append(action)
    return decided
