"""T501 (with T503): the rules engine. A rule that is on and has logic decides
on a new scoring; its decision is logged once per order, with a stable arm,
and always as a dry run until the guardrails (T502) and channels (T504).
"""

import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from fitcore.documents import check

from cfi_api import engine, rules
from cfi_api.engine import Fire, Scoring, arm, decide, fee_sensitivity
from cfi_api.store import MemoryStore

T0 = datetime(2026, 11, 28, 6, 0, tzinfo=UTC)


def scoring(
    quadrant: str = "product",
    fee: float = -0.06,
    order_id: str = "ord_1",
    customer_id: str = "cust_1",
    event_id: str = "00000000-0000-4000-8000-000000000001",
) -> Scoring:
    """A scoring as the engine reads it: quadrant and fee load's share of fit."""
    entry: dict[str, Any] = {
        "order_id": order_id,
        "quadrant": quadrant,
        "contributions": {"fit": {"x3": fee}},
    }
    return Scoring(customer_id=customer_id, event_id=event_id, entry=entry)


def consent(store: MemoryStore, customer_id: str, marketing: bool | None) -> None:
    """customer_id's stored consent: given, withdrawn, or None for never asked."""
    answer = None
    if marketing is not None:
        answer = {"marketing": marketing, "at": T0, "source": "sign_up"}
    store.documents[f"customers/{customer_id}"] = {
        "customer_id": customer_id,
        "consent": answer,
    }


def store_with(**statuses: str) -> MemoryStore:
    """The starter rules, each off but those given (rule_id with _ for -), and
    cust_1 with marketing consent."""
    store = MemoryStore()
    rules.seed(store, T0)
    for name, status in statuses.items():
        store.set_rule_status(name.replace("_", "-"), status, T0)
    consent(store, "cust_1", True)
    return store


# --- Arms (T503) ---


def test_a_customer_always_gets_the_same_arm_for_a_rule() -> None:
    assert arm("win-back", "cust_1", 0.5) == arm("win-back", "cust_1", 0.5)
    assert arm("win-back", "cust_1", 0.0) == "treatment"
    assert arm("win-back", "cust_1", 1.0) == "holdout"


def test_the_holdout_is_the_rules_share_of_customers() -> None:
    customers = [f"cust_{n}" for n in range(20_000)]
    for share in (0.1, 0.5):
        held = sum(arm("fee-sensitivity", c, share) == "holdout" for c in customers)
        assert abs(held / len(customers) - share) < 0.01


def test_lowering_the_share_only_moves_holdout_customers_to_treatment() -> None:
    customers = [f"cust_{n}" for n in range(2_000)]
    at_test = {c for c in customers if arm("win-back", c, 0.5) == "holdout"}
    at_live = {c for c in customers if arm("win-back", c, 0.1) == "holdout"}
    assert at_live < at_test


def test_each_rule_splits_customers_its_own_way() -> None:
    customers = [f"cust_{n}" for n in range(2_000)]
    one = {c for c in customers if arm("win-back", c, 0.5) == "holdout"}
    other = {c for c in customers if arm("fee-sensitivity", c, 0.5) == "holdout"}
    assert one != other


# --- Fee sensitivity ---


def test_fee_sensitivity_fires_in_product_when_fees_pull_fit_down() -> None:
    assert fee_sensitivity(scoring(fee=-0.06), {}) == Fire("voucher")
    assert fee_sensitivity(scoring(fee=-0.05), {}) == Fire("voucher")
    assert fee_sensitivity(scoring(fee=-0.049), {}) is None
    assert fee_sensitivity(scoring(fee=0.2), {}) is None
    assert fee_sensitivity(scoring(quadrant="brand", fee=-0.5), {}) is None


def test_a_condition_overrides_the_default_numbers() -> None:
    assert fee_sensitivity(scoring(fee=-0.03), {"min_fee_drag": 0.02}) is not None
    lovemark = scoring(quadrant="lovemark", fee=-0.06)
    assert fee_sensitivity(lovemark, {"quadrant": "lovemark"}) is not None


def test_a_condition_the_logic_does_not_know_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        fee_sensitivity(scoring(), {"min_fee": 0.02})


# --- Deciding ---


def test_rules_that_are_off_or_have_no_logic_decide_nothing() -> None:
    assert decide(store_with(), scoring(), T0) == []
    assert decide(store_with(service_recovery="dry_run"), scoring(), T0) == []


def test_a_rule_that_fires_is_logged_as_an_action() -> None:
    store = store_with(fee_sensitivity="dry_run")
    decided = decide(store, scoring(), T0)
    assert decided == [
        {
            "action_id": "fee-sensitivity--ord_1",
            "rule_id": "fee-sensitivity",
            "customer_id": "cust_1",
            "arm": arm("fee-sensitivity", "cust_1", 0.5),
            "channel": "voucher",
            "dry_run": True,
            "decided_at": T0,
            "outcome": None,
        }
    ]
    check("actions", decided[0])
    assert store.list_actions(10) == decided


def test_a_rule_decides_once_per_order() -> None:
    store = store_with(fee_sensitivity="dry_run")
    assert len(decide(store, scoring(), T0)) == 1
    again = scoring(event_id="00000000-0000-4000-8000-000000000002")  # rated later
    assert decide(store, again, T0) == []
    assert decide(store, scoring(), T0) == []  # a Pub/Sub retry
    assert len(decide(store, scoring(order_id="ord_2"), T0)) == 1
    assert len(store.list_actions(10)) == 2


@pytest.mark.parametrize("status", ["dry_run", "test", "live"])
def test_until_t502_and_t504_every_decision_is_a_dry_run(status: str) -> None:
    store = store_with(fee_sensitivity=status)
    assert [a["dry_run"] for a in decide(store, scoring(), T0)] == [True]


def test_a_rule_with_a_bad_condition_is_skipped_and_the_others_still_decide(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = store_with(fee_sensitivity="dry_run", win_back="dry_run")
    path = "rules/fee-sensitivity"
    store.documents[path] = {**store.documents[path], "condition": {"min_fee": 1}}
    monkeypatch.setitem(engine.LOGIC, "win-back", lambda s, c: Fire("push"))
    with caplog.at_level(logging.ERROR, logger="cfi_api.engine"):
        decided = decide(store, scoring(), T0)
    assert [a["rule_id"] for a in decided] == ["win-back"]
    assert "Rule fee-sensitivity was skipped" in caplog.text


def test_latest_actions_come_newest_first() -> None:
    store = store_with(fee_sensitivity="dry_run")
    for n in range(3):
        later = datetime(2026, 11, 28, 6, n, tzinfo=UTC)
        decide(store, scoring(order_id=f"ord_{n}"), later)
    assert [a["action_id"] for a in store.list_actions(2)] == [
        "fee-sensitivity--ord_2",
        "fee-sensitivity--ord_1",
    ]


def test_a_rule_status_change_is_checked_and_kept() -> None:
    store = store_with()
    changed = store.set_rule_status("win-back", "test", T0)
    assert (changed["status"], changed["updated_at"]) == ("test", T0)
    with pytest.raises(ValueError, match="Not a valid rules document"):
        store.set_rule_status("win-back", "on", T0)
    with pytest.raises(KeyError):
        store.set_rule_status("nothing", "off", T0)
    assert store.documents["rules/win-back"]["status"] == "test"


# --- Marketing consent (T610) ---


@pytest.mark.parametrize("marketing", [None, False], ids=["never asked", "said no"])
def test_a_marketing_rule_holds_back_without_consent(
    marketing: bool | None, caplog: pytest.LogCaptureFixture
) -> None:
    store = store_with(fee_sensitivity="dry_run")
    consent(store, "cust_1", marketing)
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        assert decide(store, scoring(), T0) == []
    assert "fee-sensitivity held back for cust_1: no marketing consent" in caplog.text
    assert store.list_actions(10) == []


def test_a_customer_saved_before_t610_has_no_consent() -> None:
    store = store_with(fee_sensitivity="dry_run")
    store.documents["customers/cust_1"] = {"customer_id": "cust_1"}  # no field
    assert decide(store, scoring(), T0) == []
    del store.documents["customers/cust_1"]  # no customer at all
    assert decide(store, scoring(), T0) == []


def test_withdrawing_consent_stops_the_next_decisions() -> None:
    store = store_with(fee_sensitivity="dry_run")
    assert len(decide(store, scoring(order_id="ord_1"), T0)) == 1
    consent(store, "cust_1", False)
    assert decide(store, scoring(order_id="ord_2"), T0) == []
    consent(store, "cust_1", True)
    assert len(decide(store, scoring(order_id="ord_3"), T0)) == 1


def test_a_service_rule_needs_no_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    store = store_with(service_recovery="dry_run", fee_sensitivity="dry_run")
    consent(store, "cust_1", None)
    monkeypatch.setitem(engine.LOGIC, "service-recovery", lambda s, c: Fire("push"))
    assert [a["rule_id"] for a in decide(store, scoring(), T0)] == ["service-recovery"]


def test_consent_is_read_once_and_only_when_a_marketing_rule_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_with(fee_sensitivity="dry_run", win_back="dry_run")
    monkeypatch.setitem(engine.LOGIC, "win-back", lambda s, c: Fire("push"))
    reads: list[str] = []
    read = store.customer_document

    def counted(customer_id: str) -> dict[str, Any] | None:
        reads.append(customer_id)
        return read(customer_id)

    monkeypatch.setattr(store, "customer_document", counted)
    assert len(decide(store, scoring(), T0)) == 2
    assert reads == ["cust_1"]
    reads.clear()
    assert decide(store, scoring(quadrant="brand", order_id="ord_9"), T0) != []
    monkeypatch.delitem(engine.LOGIC, "win-back")
    reads.clear()
    assert decide(store, scoring(quadrant="brand", order_id="ord_10"), T0) == []
    assert reads == []
