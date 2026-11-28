"""T406: the automation rules. python -m cfi_api.rules seed adds the design
doc's six starter rules, each off, and GET /rules lists them for every staff
role, by name.

These run on the memory store; test_firestore_store.py runs the store's half
against the Firestore emulator.
"""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fitcore.documents import check
from flask import Flask
from flask.testing import FlaskClient
from test_access import PROJECT, bearer, guest

from cfi_api import create_app, rules
from cfi_api.access import ACCESS, EMULATOR
from cfi_api.store import MemoryStore

T0 = datetime(2026, 11, 28, 5, 0, tzinfo=UTC)
RULES = "/api/v1/rules"
BY_NAME = [
    "Brand → Lovemark",
    "Fad protection",
    "Fee sensitivity",
    "Lovemark advocacy",
    "Service recovery",
    "Win-back",
]


def _app(**config: Any) -> Flask:
    with pytest.MonkeyPatch.context() as env:
        for name in [name for name in os.environ if name.startswith("CFI_")]:
            env.delenv(name)
        return create_app(config)


def memory(app: Flask) -> MemoryStore:
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    return store


def test_seeding_adds_the_six_starter_rules_each_off() -> None:
    store = MemoryStore()
    added = rules.seed(store, T0)
    assert added == [
        "service-recovery",
        "win-back",
        "brand-to-lovemark",
        "fad-protection",
        "fee-sensitivity",
        "lovemark-advocacy",
    ]
    found = store.list_rules()
    assert [rule["name"] for rule in found] == BY_NAME
    for rule in found:
        check("rules", rule)
        assert rule["status"] == "off"
        assert rule["holdout_share"] == 0.5
        assert rule["condition"] == rule["action"] == rule["caps"] == {}
        assert rule["updated_at"] == T0
    kinds = {rule["rule_id"]: rule["kind"] for rule in found}
    assert [rule for rule, kind in kinds.items() if kind == "service"] == [
        "service-recovery"
    ]


def test_seeding_again_adds_nothing_and_keeps_a_changed_rule() -> None:
    store = MemoryStore()
    rules.seed(store, T0)
    path = "rules/win-back"
    store.documents[path] = {**store.documents[path], "status": "dry_run"}
    del store.documents["rules/fad-protection"]
    assert rules.seed(store, T0) == ["fad-protection"]
    assert rules.seed(store, T0) == []
    assert store.documents[path]["status"] == "dry_run"
    assert len(store.list_rules()) == 6


def test_a_rule_is_added_once_and_checked_first() -> None:
    store = MemoryStore()
    document = rules.rule_document(rules.STARTER_RULES[0], T0)
    store.add_rule(document)
    with pytest.raises(ValueError, match="there already"):
        store.add_rule(document)
    with pytest.raises(ValueError, match="Not a valid rules document"):
        store.add_rule({**document, "rule_id": "other", "status": "on"})
    assert len(store.list_rules()) == 1


def test_the_list_is_a_copy() -> None:
    store = MemoryStore()
    rules.seed(store, T0)
    store.list_rules()[0]["status"] = "live"
    assert {rule["status"] for rule in store.list_rules()} == {"off"}


def test_get_rules_lists_every_rule_by_name() -> None:
    app = _app()
    client = app.test_client()
    assert client.get(RULES).get_json() == {"rules": []}
    rules.seed(memory(app), T0)
    answer = client.get(RULES)
    assert answer.status_code == 200
    found = answer.get_json()["rules"]
    assert [rule["name"] for rule in found] == BY_NAME
    assert found[4] == {
        "rule_id": "service-recovery",
        "name": "Service recovery",
        "kind": "service",
        "status": "off",
        "holdout_share": 0.5,
        "condition": {},
        "action": {},
        "caps": {},
        "updated_at": "2026-11-28T05:00:00+00:00",
    }


def test_every_staff_role_and_a_guest_read_the_rules(
    monkeypatch: pytest.MonkeyPatch, reference_model: str
) -> None:
    monkeypatch.setenv(EMULATOR, "127.0.0.1:9099")
    app = _app(
        MODEL_PATH=reference_model,
        MODEL_VERSION="t112-sim",
        AUTH="firebase",
        PROJECT=PROJECT,
    )
    client: FlaskClient = app.test_client()
    assert client.get(RULES).status_code == 401
    assert client.get(RULES, headers=bearer()).status_code == 403  # no role
    for role in ["viewer", "analyst", "admin"]:
        assert client.get(RULES, headers=bearer(role)).status_code == 200
    assert client.get(RULES, headers=guest()).status_code == 200
    assert client.post(RULES, headers=bearer("admin")).status_code == 405


def test_get_rules_is_documented() -> None:
    client = _app().test_client()
    spec = client.get("/api/v1/openapi.json").get_json()
    get = spec["paths"]["/api/v1/rules"]["get"]
    assert set(get["responses"]) == {"200", "401", "403", "default"}
    assert get["security"] == ACCESS["automations.RuleList", "GET"].security
    schema = get["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/Rules"}


def test_the_command_line_seeds_and_lists(capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore()
    assert rules.main(["seed"], store=store) == 0
    out = capsys.readouterr().out
    assert out.startswith("Added 6 starter rules: service-recovery, win-back, ")
    assert "  service-recovery   off      service    Service recovery\n" in out
    assert rules.main(["seed"], store=store) == 0
    assert "Every starter rule is there already; nothing added." in (
        capsys.readouterr().out
    )
    assert rules.main(["list"], store=store) == 0
    assert capsys.readouterr().out.count(" off ") == 6


def test_the_command_line_sets_a_status_and_lists_decisions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = MemoryStore()
    rules.seed(store, T0)
    assert rules.main(["set", "fee-sensitivity", "dry_run"], store=store) == 0
    out = capsys.readouterr().out
    assert out.startswith("fee-sensitivity is now dry_run.\n")
    assert "  fee-sensitivity    dry_run  marketing  Fee sensitivity\n" in out
    assert store.documents["rules/fee-sensitivity"]["status"] == "dry_run"
    assert rules.main(["set", "nothing", "off"], store=store) == 1
    assert "Refused: 'nothing'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        rules.main(["set", "win-back", "on"], store=store)
    capsys.readouterr()

    action = {
        "action_id": "fee-sensitivity--ord_1",
        "rule_id": "fee-sensitivity",
        "customer_id": "cust_1",
        "arm": "holdout",
        "channel": "voucher",
        "dry_run": True,
        "decided_at": T0,
        "cost": 4_000,
        "send_at": T0,
        "outcome": None,
    }
    assert store.add_action(action)
    assert not store.add_action(action)
    assert rules.main(["actions"], store=store) == 0
    assert capsys.readouterr().out == (
        "  2026-11-28 05:00:00  fee-sensitivity    cust_1               holdout   voucher"
        " 40.00 baht\n"
    )


def test_the_command_line_pauses_resumes_and_sets_the_guardrails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = MemoryStore()
    assert rules.main(["guardrails"], store=store) == 0
    out = capsys.readouterr().out
    assert out.startswith("Automations: on (defaults: nothing has been set)\n")
    assert "  frequency cap  2 marketing actions per customer in 7 days\n" in out
    assert "  quiet hours    21:00 to 08:00 (Asia/Bangkok)\n" in out
    assert "  daily budget   5,000.00 baht; spent on " in out
    assert store.automation_settings() is None  # showing writes nothing

    assert rules.main(["pause"], store=store) == 0
    assert capsys.readouterr().out.startswith(
        "Automations: paused (the global kill switch is on)\n"
    )
    assert store.automation_settings()["enabled"] is False  # type: ignore[index]
    argv = ["guardrails", "--cap", "1", "--cap-days", "3", "--quiet", "22:00-07:00"]
    assert rules.main([*argv, "--budget", "1000"], store=store) == 0
    out = capsys.readouterr().out
    assert "paused" in out  # changing settings keeps the switch as it was
    assert "  frequency cap  1 marketing actions per customer in 3 days\n" in out
    assert "  quiet hours    22:00 to 07:00 (Asia/Bangkok)\n" in out
    assert "  daily budget   1,000.00 baht;" in out
    assert rules.main(["resume"], store=store) == 0
    assert capsys.readouterr().out.startswith("Automations: on\n")
    settings = store.automation_settings()
    assert settings is not None
    assert settings["frequency_cap"] == {"actions": 1, "days": 3}
    assert settings["daily_budget"] == 100_000

    assert rules.main(["guardrails", "--cap", "0"], store=store) == 1
    assert "Refused" in capsys.readouterr().err
    assert rules.main(["guardrails", "--quiet", "25:00-07:00"], store=store) == 1
    capsys.readouterr()
    with pytest.raises(SystemExit):
        rules.main(["guardrails", "--quiet", "late"], store=store)
    assert store.automation_settings() == settings  # nothing refused was kept


def test_actions_say_what_they_cost_and_when_they_wait(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = MemoryStore()
    action = {
        "action_id": "fee-sensitivity--ord_1",
        "rule_id": "fee-sensitivity",
        "customer_id": "cust_1",
        "arm": "treatment",
        "channel": "voucher",
        "dry_run": True,
        "decided_at": T0,
        "cost": 4_000,
        "send_at": T0.replace(hour=1, minute=0) + timedelta(days=1),
        "outcome": None,
    }
    store.add_action(action)
    assert rules.main(["actions"], store=store) == 0
    assert capsys.readouterr().out == (
        "  2026-11-28 05:00:00  fee-sensitivity    cust_1               treatment voucher"
        " 40.00 baht, waits until 2026-11-29 01:00 UTC\n"
    )


def test_the_command_line_needs_a_project(capsys: pytest.CaptureFixture[str]) -> None:
    assert rules.main(["list"]) == 2
    assert "--project" in capsys.readouterr().err


def test_the_command_line_runs_without_a_double_import_warning() -> None:
    # cfi_api never imports the command line, so python -m cfi_api.rules
    # loads it once.
    command = ["-W", "error::RuntimeWarning", "-m", "cfi_api.rules", "list"]
    run = subprocess.run(
        [sys.executable, *command], capture_output=True, text=True, check=False
    )
    assert (run.returncode, run.stderr.strip()) == (
        2,
        "Give --project, the project with the rules.",
    )
