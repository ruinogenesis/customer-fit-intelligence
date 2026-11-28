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
"""

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
# fee-free delivery is worth about a typical delivery fee. A rule not here
# sends messages, which cost nothing.
ACTIONS: dict[str, dict[str, int]] = {"fee-sensitivity": {"value": 4_000}}


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
    scoring: Scoring,
    fire: Fire,
    worth: int,
    now: datetime,
    send_at: datetime,
) -> dict[str, Any]:
    """The document for actions/{action_id}, checked; always a dry run until
    the channels (T504)."""
    document = {
        "action_id": action_id(rule["rule_id"], scoring.entry["order_id"]),
        "rule_id": rule["rule_id"],
        "customer_id": scoring.customer_id,
        "arm": arm(rule["rule_id"], scoring.customer_id, rule["holdout_share"]),
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

    def add_action(self, document: dict[str, Any], *, day: str, budget: int) -> bool:
        """Add actions/{action_id} and count its spend on day; False if it is
        there already. Raises OverBudget past budget (cfi_api.store)."""
        ...


def _held(rule: Mapping[str, Any], scoring: Scoring, why: str) -> None:
    logger.info("%s held back for %s: %s", rule["rule_id"], scoring.customer_id, why)


def decide(store: Automations, scoring: Scoring, now: datetime) -> list[dict[str, Any]]:
    """Ask every rule that is on and has logic about scoring, and log each
    that fires and passes the guardrails. The decisions logged now, in rule
    order: none for a rule that had decided on this order already.

    A rule whose condition or action the logic refuses is skipped and logged
    as an error, so one bad rule never stops the others.
    """
    guard = Guardrails.from_document(store.automation_settings())
    if not guard.enabled:
        logger.info(
            "Automations are paused: nothing decided for %s", scoring.customer_id
        )
        return []
    rules = store.list_rules()
    marketing = {rule["rule_id"] for rule in rules if rule["kind"] == MARKETING}
    day, quiet = guard.local_day(now), guard.quiet_until(now)
    decided = []
    # Read once, and only when a marketing rule fires.
    consent: bool | None = None
    recent: int | None = None  # the customer's marketing decisions in the cap
    known: set[str] = set()  # their actions in the cap's days, by action_id
    for rule in rules:
        logic = LOGIC.get(rule["rule_id"])
        if rule["status"] == OFF or logic is None:
            continue
        try:
            fire = logic(scoring, rule["condition"])
            worth = cost(rule)
        except ValueError:
            logger.exception("Rule %s was skipped: its numbers", rule["rule_id"])
            continue
        if fire is None:
            continue
        send_at = now
        if rule["kind"] == MARKETING:
            if consent is None:
                consent = consents(store.customer_document(scoring.customer_id))
            if not consent:
                _held(rule, scoring, "no marketing consent")
                continue
            if recent is None:
                since = now - timedelta(days=guard.cap_days)
                found = store.customer_actions(scoring.customer_id, since)
                recent = sum(action["rule_id"] in marketing for action in found)
                known = {action["action_id"] for action in found}
            if action_id(rule["rule_id"], scoring.entry["order_id"]) in known:
                continue  # decided on this order already: nothing to hold back
            if recent >= guard.cap_actions:
                cap = f"{guard.cap_actions} in {guard.cap_days} days"
                _held(rule, scoring, f"frequency cap ({cap})")
                continue
            send_at = quiet or now
        document = action_document(rule, scoring, fire, worth, now, send_at)
        try:
            added = store.add_action(document, day=day, budget=guard.daily_budget)
        except OverBudget:
            _held(rule, scoring, f"daily budget ({day})")
            continue
        if added:
            decided.append(document)
            if rule["kind"] == MARKETING and recent is not None:
                recent += 1
    return decided
