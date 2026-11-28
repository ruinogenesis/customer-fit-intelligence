"""The automation rules' admin commands (T406): python -m cfi_api.rules.

    python -m cfi_api.rules --project P seed
    python -m cfi_api.rules --project P list

seed writes the design doc's six starter rules to rules/{rule_id}, each off
(its kill switch on), with the half of its customers its first test will
hold out: nothing evaluates a rule yet. Their condition, action and caps
stay empty until the rules engine (T501) and the guardrails (T502) give
them a form. A rule that is there already is left as it is, so seeding
twice adds nothing and never undoes a change. It runs as you, with
Application Default Credentials, as cfi_api.roles does.

The API never imports this module: GET /rules (cfi_api.automations) reads
the rules through the store.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from fitcore.documents import check

OFF = "off"
FIRST_HOLDOUT = 0.5  # a 50/50 test, the first step after dry run


@dataclass(frozen=True)
class StarterRule:
    rule_id: str
    name: str
    kind: str  # service rules skip the marketing consent and frequency cap


# The design doc's starter rules, in its order. Only service recovery counts
# as service; every other rule is marketing.
STARTER_RULES: tuple[StarterRule, ...] = (
    StarterRule("service-recovery", "Service recovery", "service"),
    StarterRule("win-back", "Win-back", "marketing"),
    StarterRule("brand-to-lovemark", "Brand → Lovemark", "marketing"),
    StarterRule("fad-protection", "Fad protection", "marketing"),
    StarterRule("fee-sensitivity", "Fee sensitivity", "marketing"),
    StarterRule("lovemark-advocacy", "Lovemark advocacy", "marketing"),
)


def rule_document(rule: StarterRule, now: datetime) -> dict[str, Any]:
    """The document for rules/{rule_id}: off, checked against its schema."""
    document = {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "kind": rule.kind,
        "status": OFF,
        "holdout_share": FIRST_HOLDOUT,
        "condition": {},
        "action": {},
        "caps": {},
        "updated_at": now,
    }
    check("rules", document)
    return document


class Rules(Protocol):
    """What the admin commands need from a store."""

    def list_rules(self) -> list[dict[str, Any]]: ...

    def add_rule(self, document: dict[str, Any]) -> None: ...


def seed(store: Rules, now: datetime) -> list[str]:
    """Add the starter rules that are not there yet; the ids added, in order."""
    there = {rule["rule_id"] for rule in store.list_rules()}
    added = []
    for rule in STARTER_RULES:
        if rule.rule_id not in there:
            store.add_rule(rule_document(rule, now))
            added.append(rule.rule_id)
    return added


def main(argv: Sequence[str] | None = None, *, store: Rules | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfi_api.rules", description="The automation rules (T406)."
    )
    parser.add_argument("--project", help="the Google Cloud project")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed", help="add the starter rules that are not there yet")
    commands.add_parser("list", help="every rule, by name")
    args = parser.parse_args(argv)

    if store is None:
        if not args.project:
            print("Give --project, the project with the rules.", file=sys.stderr)
            return 2
        from google.cloud import firestore

        from cfi_api.firestore_store import FirestoreStore

        store = FirestoreStore(firestore.Client(project=args.project))

    try:
        if args.command == "seed":
            added = seed(store, datetime.now(UTC))
            if added:
                print(f"Added {len(added)} starter rules: {', '.join(added)}.")
            else:
                print("Every starter rule is there already; nothing added.")
        for rule in store.list_rules():
            status, kind = rule["status"], rule["kind"]
            print(f"  {rule['rule_id']:18} {status:8} {kind:10} {rule['name']}")
    except ValueError as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
