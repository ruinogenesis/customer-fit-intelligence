"""The automation rules' admin commands (T406, T501, T502): python -m cfi_api.rules.

    python -m cfi_api.rules --project P seed
    python -m cfi_api.rules --project P list
    python -m cfi_api.rules --project P set RULE STATUS
    python -m cfi_api.rules --project P actions [--limit N]
    python -m cfi_api.rules --project P pause | resume
    python -m cfi_api.rules --project P guardrails [--cap N] [--cap-days D]
        [--quiet HH:MM-HH:MM] [--budget BAHT]

seed writes the design doc's six starter rules to rules/{rule_id}, each off
(its kill switch on), with the half of its customers its first test will
hold out. Their condition and action maps start empty: a rule's logic
(cfi_api.engine) has its own default numbers, which they override. A rule
that is there already is left as it is, so seeding twice adds nothing and
never undoes a change.

set changes a rule's status: off, dry_run, test or live. Until the channels
(T504) exist, the engine logs every decision as a dry run whatever the
status. actions lists the engine's latest decisions (T501).

pause and resume flip the global kill switch, and guardrails shows or
changes the settings every rule shares (settings/automations, T502); a
setting not given keeps its value, or the default before anything was set.
It all runs as you, with Application Default Credentials, as cfi_api.roles
does.

The API never imports this module: GET /rules (cfi_api.automations) reads
the rules through the store.
"""

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from fitcore.documents import RULE_STATUSES, check

from cfi_api.engine import OFF, Guardrails

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

    def set_rule_status(
        self, rule_id: str, status: str, now: datetime
    ) -> dict[str, Any]: ...

    def list_actions(self, limit: int) -> list[dict[str, Any]]: ...

    def automation_settings(self) -> dict[str, Any] | None: ...

    def set_automation_settings(self, document: dict[str, Any]) -> None: ...

    def spent_on(self, day: str) -> int: ...


def seed(store: Rules, now: datetime) -> list[str]:
    """Add the starter rules that are not there yet; the ids added, in order."""
    there = {rule["rule_id"] for rule in store.list_rules()}
    added = []
    for rule in STARTER_RULES:
        if rule.rule_id not in there:
            store.add_rule(rule_document(rule, now))
            added.append(rule.rule_id)
    return added


def _quiet(text: str) -> tuple[str, str]:
    found = re.fullmatch(r"(\d\d:\d\d)-(\d\d:\d\d)", text)
    if found is None:
        raise argparse.ArgumentTypeError(f"expected HH:MM-HH:MM, got {text!r}")
    return found.group(1), found.group(2)


def _baht(satang: int) -> str:
    return f"{satang / 100:,.2f} baht"


def show_guardrails(store: Rules, now: datetime) -> None:
    """Print the guardrails every rule shares, and what today has spent."""
    found = store.automation_settings()
    guard = Guardrails.from_document(found)
    state = "on" if guard.enabled else "paused (the global kill switch is on)"
    note = "" if found is not None else " (defaults: nothing has been set)"
    day = guard.local_day(now)
    print(f"Automations: {state}{note}")
    print(
        f"  frequency cap  {guard.cap_actions} marketing actions per customer"
        f" in {guard.cap_days} days"
    )
    print(
        f"  quiet hours    {guard.quiet_start} to {guard.quiet_end} ({guard.time_zone})"
    )
    print(
        f"  daily budget   {_baht(guard.daily_budget)};"
        f" spent on {day}: {_baht(store.spent_on(day))}"
    )


def main(argv: Sequence[str] | None = None, *, store: Rules | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfi_api.rules", description="The automation rules (T406)."
    )
    parser.add_argument("--project", help="the Google Cloud project")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed", help="add the starter rules that are not there yet")
    commands.add_parser("list", help="every rule, by name")
    status = commands.add_parser("set", help="change a rule's status")
    status.add_argument("rule_id")
    status.add_argument("status", choices=RULE_STATUSES)
    actions = commands.add_parser("actions", help="the engine's latest decisions")
    actions.add_argument("--limit", type=int, default=10)
    commands.add_parser("pause", help="the global kill switch: no rule decides")
    commands.add_parser("resume", help="let the rules decide again")
    guardrails = commands.add_parser(
        "guardrails", help="show or change the settings every rule shares"
    )
    guardrails.add_argument("--cap", type=int, help="marketing actions per customer")
    guardrails.add_argument("--cap-days", type=int, help="over this many days")
    guardrails.add_argument("--quiet", type=_quiet, help="quiet hours, HH:MM-HH:MM")
    guardrails.add_argument("--budget", type=int, help="whole baht per local day")
    args = parser.parse_args(argv)

    if store is None:
        if not args.project:
            print("Give --project, the project with the rules.", file=sys.stderr)
            return 2
        from google.cloud import firestore

        from cfi_api.firestore_store import FirestoreStore

        store = FirestoreStore(firestore.Client(project=args.project))

    try:
        now = datetime.now(UTC)
        if args.command == "actions":
            for action in store.list_actions(args.limit):
                line = (
                    f"  {action['decided_at']:%Y-%m-%d %H:%M:%S}  {action['rule_id']:18}"
                    f" {action['customer_id']:20} {action['arm']:9} {action['channel']}"
                )
                if action.get("cost"):  # decisions from before T502 have none
                    line += f" {_baht(action['cost'])}"
                if action.get("send_at", action["decided_at"]) > action["decided_at"]:
                    line += f", waits until {action['send_at']:%Y-%m-%d %H:%M} UTC"
                print(line)
            return 0
        if args.command in ("pause", "resume", "guardrails"):
            guard = Guardrails.from_document(store.automation_settings())
            if args.command == "guardrails":
                quiet = args.quiet or (guard.quiet_start, guard.quiet_end)
                guard = replace(
                    guard,
                    cap_actions=args.cap if args.cap is not None else guard.cap_actions,
                    cap_days=args.cap_days
                    if args.cap_days is not None
                    else guard.cap_days,
                    quiet_start=quiet[0],
                    quiet_end=quiet[1],
                    daily_budget=(
                        args.budget * 100
                        if args.budget is not None
                        else guard.daily_budget
                    ),
                )
                changed = any(
                    value is not None
                    for value in (args.cap, args.cap_days, args.quiet, args.budget)
                )
            else:
                guard = replace(guard, enabled=args.command == "resume")
                changed = True
            if changed:
                store.set_automation_settings(guard.document(now))
            show_guardrails(store, now)
            return 0
        if args.command == "seed":
            added = seed(store, datetime.now(UTC))
            if added:
                print(f"Added {len(added)} starter rules: {', '.join(added)}.")
            else:
                print("Every starter rule is there already; nothing added.")
        elif args.command == "set":
            store.set_rule_status(args.rule_id, args.status, datetime.now(UTC))
            print(f"{args.rule_id} is now {args.status}.")
        for rule in store.list_rules():
            status, kind = rule["status"], rule["kind"]
            print(f"  {rule['rule_id']:18} {status:8} {kind:10} {rule['name']}")
    except (ValueError, KeyError) as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
