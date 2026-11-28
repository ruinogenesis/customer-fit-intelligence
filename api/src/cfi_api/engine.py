"""The rules engine (T501): which rules fire on a new scoring, and their decisions.

The rules service (cfi_api.rules_service) hands decide() every new score
entry. Each rule that is not off and has its logic here (LOGIC) is asked
whether it fires; the numbers it fires on come from its condition map, over
the logic's own defaults. Each rule that fires is logged to
actions/{action_id} with the customer's arm for it (T503, folded into T501):
a stable hash of rule and customer puts the rule's holdout_share of
customers in the holdout, where the decision is logged and nothing is done.

A marketing rule decides only for a customer who holds marketing consent
now (T610, the first guardrail): one never asked, or who said no or took it
back, is not part of the rule at all, so nothing is logged for them. Service
rules need no consent.

Until the other guardrails (T502) and the channels (T504) exist, every
decision is logged as a dry run, whatever the rule's status: nothing is
ever sent.

A rule decides at most once per order. An order is scored again as it is
delivered and rated, and Pub/Sub may send one scoring twice; action_id is
the rule and the order, so neither logs a second decision.
"""

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from fitcore.documents import check

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


def numbers(
    condition: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, Any]:
    """The rule's numbers: its condition over the logic's defaults.

    Raises ValueError for a name the logic does not know, so a typo in a
    condition is refused instead of ignored.
    """
    unknown = set(condition) - set(defaults)
    if unknown:
        raise ValueError(
            f"unknown condition {sorted(unknown)}; known {sorted(defaults)}"
        )
    return {**defaults, **condition}


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
    rule: Mapping[str, Any], scoring: Scoring, fire: Fire, now: datetime
) -> dict[str, Any]:
    """The document for actions/{action_id}, checked; always a dry run (T501)."""
    document = {
        "action_id": action_id(rule["rule_id"], scoring.entry["order_id"]),
        "rule_id": rule["rule_id"],
        "customer_id": scoring.customer_id,
        "arm": arm(rule["rule_id"], scoring.customer_id, rule["holdout_share"]),
        "channel": fire.channel,
        "dry_run": True,
        "decided_at": now,
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

    def list_rules(self) -> list[dict[str, Any]]: ...

    def customer_document(self, customer_id: str) -> dict[str, Any] | None: ...

    def add_action(self, document: dict[str, Any]) -> bool:
        """Add actions/{action_id}; False if it is there already."""
        ...


def decide(store: Automations, scoring: Scoring, now: datetime) -> list[dict[str, Any]]:
    """Ask every rule that is on and has logic about scoring; log each that
    fires. The decisions logged now, in rule order: none for a rule that had
    decided on this order already.

    A rule whose condition the logic refuses is skipped and logged as an
    error, so one bad rule never stops the others. A marketing rule that
    fires for a customer without marketing consent logs nothing (T610).
    """
    decided = []
    consent: bool | None = None  # read once, and only if a marketing rule fires
    for rule in store.list_rules():
        logic = LOGIC.get(rule["rule_id"])
        if rule["status"] == OFF or logic is None:
            continue
        try:
            fire = logic(scoring, rule["condition"])
        except ValueError:
            logger.exception("Rule %s was skipped: its condition", rule["rule_id"])
            continue
        if fire is None:
            continue
        if rule["kind"] == MARKETING:
            if consent is None:
                consent = consents(store.customer_document(scoring.customer_id))
            if not consent:
                logger.info(
                    "%s held back for %s: no marketing consent",
                    rule["rule_id"],
                    scoring.customer_id,
                )
                continue
        document = action_document(rule, scoring, fire, now)
        if store.add_action(document):
            decided.append(document)
    return decided
