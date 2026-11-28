"""T501 (with T503): the rules engine. A rule that is on and has logic decides
on a new scoring; its decision is logged once per order, with a stable arm,
and always as a dry run until the guardrails (T502) and channels (T504).
"""

import dataclasses
import logging
from datetime import UTC, datetime, timedelta
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


def guard(store: MemoryStore, **changes: Any) -> None:
    """Change the guardrails' settings from their defaults (T502)."""
    settings = engine.Guardrails.from_document(store.automation_settings())
    store.set_automation_settings(dataclasses.replace(settings, **changes).document(T0))


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


def fee(new: Scoring, condition: dict[str, Any]) -> Fire | None:
    """fee_sensitivity on new, with a history it never reads."""
    return fee_sensitivity(new, condition, engine.History(MemoryStore(), "cust_1"))


def test_fee_sensitivity_fires_in_product_when_fees_pull_fit_down() -> None:
    assert fee(scoring(fee=-0.06), {}) == Fire("voucher")
    assert fee(scoring(fee=-0.05), {}) == Fire("voucher")
    assert fee(scoring(fee=-0.049), {}) is None
    assert fee(scoring(fee=0.2), {}) is None
    assert fee(scoring(quadrant="brand", fee=-0.5), {}) is None


def test_a_condition_overrides_the_default_numbers() -> None:
    assert fee(scoring(fee=-0.03), {"min_fee_drag": 0.02}) is not None
    lovemark = scoring(quadrant="lovemark", fee=-0.06)
    assert fee(lovemark, {"quadrant": "lovemark"}) is not None


def test_a_condition_the_logic_does_not_know_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown"):
        fee(scoring(), {"min_fee": 0.02})


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
            "cost": 4_000,
            "send_at": T0,  # 13:00 in Bangkok: not quiet hours
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
    monkeypatch.setitem(engine.LOGIC, "win-back", lambda s, c, h: Fire("push"))
    with caplog.at_level(logging.ERROR, logger="cfi_api.engine"):
        decided = decide(store, scoring(), T0)
    assert [a["rule_id"] for a in decided] == ["win-back"]
    assert "Rule fee-sensitivity was skipped" in caplog.text


def test_latest_actions_come_newest_first() -> None:
    store = store_with(fee_sensitivity="dry_run")
    guard(store, cap_actions=10)
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
    monkeypatch.setitem(engine.LOGIC, "service-recovery", lambda s, c, h: Fire("push"))
    assert [a["rule_id"] for a in decide(store, scoring(), T0)] == ["service-recovery"]


def test_consent_is_read_once_and_only_when_a_marketing_rule_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_with(fee_sensitivity="dry_run", win_back="dry_run")
    guard(store, cap_actions=10)
    monkeypatch.setitem(engine.LOGIC, "win-back", lambda s, c, h: Fire("push"))
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


# --- The guardrails (T502): each one blocks ---

BANGKOK_EVENING = datetime(2026, 11, 28, 15, 0, tzinfo=UTC)  # 22:00 in Bangkok
BANGKOK_NIGHT = datetime(2026, 11, 28, 20, 30, tzinfo=UTC)  # 03:30 next day
EIGHT_AM = datetime(2026, 11, 29, 1, 0, tzinfo=UTC)  # 08:00 on the 29th


def customers_in(
    arm_name: str, count: int, rule_id: str = "fee-sensitivity"
) -> list[str]:
    """The first count customers the rule puts in arm_name at a 0.5 share."""
    found = (f"cust_{n}" for n in range(1_000))
    return [c for c in found if arm(rule_id, c, 0.5) == arm_name][:count]


def test_with_no_settings_the_defaults_hold() -> None:
    defaults = engine.Guardrails()
    assert engine.Guardrails.from_document(None) == defaults
    assert (defaults.cap_actions, defaults.cap_days) == (2, 7)
    assert (defaults.quiet_start, defaults.quiet_end) == ("21:00", "08:00")
    assert defaults.daily_budget == 500_000
    assert engine.Guardrails.from_document(defaults.document(T0)) == defaults


def test_the_global_kill_switch_stops_every_rule(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = store_with(fee_sensitivity="live", service_recovery="live")
    monkeypatch.setitem(engine.LOGIC, "service-recovery", lambda s, c, h: Fire("push"))
    guard(store, enabled=False)
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        assert decide(store, scoring(), T0) == []
    assert "Automations are paused: nothing decided for cust_1" in caplog.text
    reads = []
    monkeypatch.setattr(store, "list_rules", lambda: reads.append(1) or [])
    decide(store, scoring(order_id="ord_2"), T0)
    assert reads == []  # paused, it reads no rule at all
    monkeypatch.undo()
    monkeypatch.setitem(engine.LOGIC, "service-recovery", lambda s, c, h: Fire("push"))
    guard(store, enabled=True)
    assert len(decide(store, scoring(), T0)) == 2


def test_a_rules_own_kill_switch_stops_only_that_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_with(fee_sensitivity="off", service_recovery="live")
    monkeypatch.setitem(engine.LOGIC, "service-recovery", lambda s, c, h: Fire("push"))
    assert [a["rule_id"] for a in decide(store, scoring(), T0)] == ["service-recovery"]


def test_the_frequency_cap_holds_back_the_next_marketing_decision(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = store_with(fee_sensitivity="dry_run", win_back="dry_run")
    monkeypatch.setitem(engine.LOGIC, "win-back", lambda s, c, h: Fire("push"))
    assert len(decide(store, scoring(order_id="ord_1"), T0)) == 2  # both rules
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        assert decide(store, scoring(order_id="ord_2"), T0 + timedelta(days=6)) == []
    assert "held back for cust_1: frequency cap (2 in 7 days)" in caplog.text
    # Seven days on, the first two no longer count.
    later = decide(store, scoring(order_id="ord_3"), T0 + timedelta(days=7, minutes=1))
    assert len(later) == 2


def test_an_order_scored_again_is_not_held_back_by_its_own_decision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = store_with(fee_sensitivity="dry_run")
    guard(store, cap_actions=1)
    assert len(decide(store, scoring(order_id="ord_1"), T0)) == 1
    rated = scoring(order_id="ord_1", event_id="00000000-0000-4000-8000-000000000009")
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        assert decide(store, rated, T0) == []
        assert decide(store, scoring(order_id="ord_2"), T0) == []
    assert caplog.text.count("frequency cap") == 1  # for ord_2 only


def test_the_cap_counts_both_arms_and_spares_service_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_with(fee_sensitivity="dry_run", service_recovery="dry_run")
    guard(store, cap_actions=1)
    monkeypatch.setitem(engine.LOGIC, "service-recovery", lambda s, c, h: Fire("push"))
    held_out = customers_in("holdout", 1)[0]
    consent(store, held_out, True)
    first = decide(store, scoring(customer_id=held_out, order_id="ord_1"), T0)
    assert {a["arm"] for a in first if a["rule_id"] == "fee-sensitivity"} == {"holdout"}
    second = decide(store, scoring(customer_id=held_out, order_id="ord_2"), T0)
    assert [a["rule_id"] for a in second] == ["service-recovery"]  # exempt


def test_the_daily_budget_stops_both_arms_once_it_would_be_passed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = store_with(fee_sensitivity="dry_run")
    guard(store, daily_budget=10_000, cap_actions=10)  # 100 baht; vouchers are 40
    treated, held = customers_in("treatment", 3), customers_in("holdout", 2)
    for customer in treated + held:
        consent(store, customer, True)

    def decide_for(customer: str, at: datetime = T0) -> list[str]:
        new = scoring(customer_id=customer, order_id=f"ord_{customer}_{at:%d}")
        return [action["arm"] for action in decide(store, new, at)]

    assert decide_for(held[0]) == ["holdout"]
    assert store.spent_on("2026-11-28") == 0  # the holdout spends nothing
    assert decide_for(treated[0]) + decide_for(treated[1]) == ["treatment"] * 2
    assert store.spent_on("2026-11-28") == 8_000
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        assert decide_for(treated[2]) == []  # 80 + 40 > 100 baht
        assert decide_for(held[1]) == []  # nor a holdout, so the arms stay alike
    assert caplog.text.count("daily budget (2026-11-28)") == 2
    assert store.spent_on("2026-11-28") == 8_000
    assert decide_for(treated[2], T0 + timedelta(days=1)) == ["treatment"]
    assert store.spent_on("2026-11-29") == 4_000


def test_a_message_costs_nothing_so_the_budget_never_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_with(brand_to_lovemark="dry_run")
    guard(store, daily_budget=0)
    monkeypatch.setitem(engine.LOGIC, "brand-to-lovemark", lambda s, c, h: Fire("push"))
    decided = decide(store, scoring(), T0)
    assert [(a["rule_id"], a["cost"]) for a in decided] == [("brand-to-lovemark", 0)]


def test_a_rules_action_map_sets_what_it_costs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = store_with(fee_sensitivity="dry_run")
    path = "rules/fee-sensitivity"
    store.documents[path] = {**store.documents[path], "action": {"value": 2_500}}
    assert [a["cost"] for a in decide(store, scoring(), T0)] == [2_500]
    for bad in ({"value": -1}, {"value": 25.0}, {"value": True}, {"worth": 1}):
        store.documents[path] = {**store.documents[path], "action": bad}
        with caplog.at_level(logging.ERROR, logger="cfi_api.engine"):
            assert decide(store, scoring(order_id=f"ord_{bad}"), T0) == []
    assert caplog.text.count("Rule fee-sensitivity was skipped") == 4


@pytest.mark.parametrize(
    ("now", "send_at"),
    [
        (T0, T0),  # 13:00 in Bangkok
        (BANGKOK_EVENING, EIGHT_AM),
        (BANGKOK_NIGHT, EIGHT_AM),
        (EIGHT_AM, EIGHT_AM),  # 08:00 sharp: quiet hours are over
        (
            datetime(2026, 11, 28, 13, 59, tzinfo=UTC),
            datetime(2026, 11, 28, 13, 59, tzinfo=UTC),
        ),
    ],
    ids=["afternoon", "evening", "night", "eight sharp", "20:59"],
)
def test_quiet_hours_hold_a_marketing_action_until_they_end(
    now: datetime, send_at: datetime
) -> None:
    store = store_with(fee_sensitivity="dry_run")
    assert [a["send_at"] for a in decide(store, scoring(), now)] == [send_at]


def test_service_actions_go_at_once_even_in_quiet_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_with(service_recovery="dry_run")
    monkeypatch.setitem(engine.LOGIC, "service-recovery", lambda s, c, h: Fire("push"))
    assert [a["send_at"] for a in decide(store, scoring(), BANGKOK_EVENING)] == [
        BANGKOK_EVENING
    ]


def test_quiet_hours_within_one_day_and_none_at_all() -> None:
    afternoon = engine.Guardrails(quiet_start="13:00", quiet_end="15:00")
    assert afternoon.quiet_until(T0) == datetime(2026, 11, 28, 8, 0, tzinfo=UTC)
    assert afternoon.quiet_until(T0 + timedelta(hours=3)) is None  # 16:00
    assert (
        engine.Guardrails(quiet_start="21:00", quiet_end="21:00").quiet_until(
            BANGKOK_EVENING
        )
        is None
    )


# --- The daily sweep (T505) ---


def quiet(
    store: MemoryStore,
    customer_id: str = "cust_1",
    days: float = 14.0,
    churn: float | None = 0.10,
    order_id: str | None = None,
    marketing: bool | None = True,
) -> None:
    """customer_id's latest order: order_id (ord_ and their id, unless given),
    placed days before T0, with its churn risk; and their consent, as
    consent() takes it."""
    consent(store, customer_id, marketing)
    store.documents[f"customers/{customer_id}"].update(
        latest_order_id=order_id or f"ord_{customer_id}",
        latest_placed_at=T0 - timedelta(days=days),
        churn_risk=churn,
    )


def test_win_back_fires_for_a_risky_customer_quiet_for_14_days() -> None:
    store = MemoryStore()
    quiet(store, "cust_quiet", days=14, churn=0.10)
    quiet(store, "cust_riskier", days=30, churn=0.40)
    quiet(store, "cust_recent", days=13.9, churn=0.90)
    quiet(store, "cust_safe", days=30, churn=0.09)
    quiet(store, "cust_unscored", days=30, churn=None)
    consent(store, "cust_new", True)  # signed up, never ordered
    found = engine.win_back(store, {}, T0)
    assert [(c.customer["customer_id"], c.order_id, c.fire) for c in found] == [
        ("cust_riskier", "ord_cust_riskier", Fire("voucher")),
        ("cust_quiet", "ord_cust_quiet", Fire("voucher")),
    ]  # riskiest first, so the daily budget goes to them first


def test_win_backs_condition_sets_its_numbers() -> None:
    store = MemoryStore()
    quiet(store, days=20, churn=0.05)
    assert engine.win_back(store, {}, T0) == []
    assert len(engine.win_back(store, {"min_churn": 0.05}, T0)) == 1
    assert engine.win_back(store, {"min_churn": 0.05, "quiet_days": 21}, T0) == []
    with pytest.raises(ValueError, match=r"unknown \['days'\]"):
        engine.win_back(store, {"days": 7}, T0)


def test_the_sweep_logs_a_win_back_once_per_quiet_spell() -> None:
    store = store_with(win_back="dry_run")
    quiet(store)
    decided = engine.sweep(store, T0)
    assert decided == [
        {
            "action_id": "win-back--ord_cust_1",
            "rule_id": "win-back",
            "customer_id": "cust_1",
            "arm": arm("win-back", "cust_1", 0.5),
            "channel": "voucher",
            "dry_run": True,
            "decided_at": T0,
            "cost": 5_000,
            "send_at": T0,
            "outcome": None,
        }
    ]
    assert store.list_actions(10) == decided
    # Every sweep sees them while they stay quiet; none decides again.
    for day in (1, 8, 30):
        assert engine.sweep(store, T0 + timedelta(days=day)) == []
    # They order again 10 days after T0: 14 days after that, a new spell.
    quiet(store, days=-10, order_id="ord_2")
    assert engine.sweep(store, T0 + timedelta(days=23)) == []
    again = engine.sweep(store, T0 + timedelta(days=24))
    assert [a["action_id"] for a in again] == ["win-back--ord_2"]


def test_the_sweep_takes_consent_from_the_customer_it_found(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = store_with(win_back="dry_run")
    quiet(store, "cust_1")
    quiet(store, "cust_2", marketing=False)
    quiet(store, "cust_3", marketing=None)
    monkeypatch.setattr(store, "customer_document", lambda c: pytest.fail(c))
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        decided = engine.sweep(store, T0)
    assert [a["customer_id"] for a in decided] == ["cust_1"]
    assert caplog.text.count("no marketing consent") == 2


def test_the_sweep_goes_through_the_same_guardrails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = store_with(win_back="dry_run", fee_sensitivity="dry_run")
    quiet(store)
    guard(store, enabled=False)
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        assert engine.sweep(store, T0) == []
    assert "Automations are paused: nothing decided for the daily sweep" in caplog.text
    guard(store, enabled=True)
    # Two fee-sensitivity decisions this week: win-back waits.
    decide(store, scoring(order_id="ord_a"), T0 - timedelta(days=2))
    decide(store, scoring(order_id="ord_b"), T0 - timedelta(days=1))
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        assert engine.sweep(store, T0) == []
    assert "win-back held back for cust_1: frequency cap (2 in 7 days)" in caplog.text
    # A week on, the cap is clear, but the rule's own kill switch is on.
    store.set_rule_status("win-back", "off", T0)
    assert engine.sweep(store, T0 + timedelta(days=8)) == []
    store.set_rule_status("win-back", "dry_run", T0)
    assert len(engine.sweep(store, T0 + timedelta(days=8))) == 1


def test_the_sweep_spends_the_daily_budget_on_the_riskiest_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = store_with(win_back="dry_run")
    guard(store, daily_budget=10_000)  # 100 baht: two 50-baht vouchers
    treated = customers_in("treatment", 3, "win-back")
    for customer, churn in zip(treated, (0.2, 0.6, 0.4), strict=True):
        quiet(store, customer, churn=churn)
    with caplog.at_level(logging.INFO, logger="cfi_api.engine"):
        decided = engine.sweep(store, T0)
    assert [a["customer_id"] for a in decided] == [treated[1], treated[2]]
    assert store.spent_on("2026-11-28") == 10_000
    assert f"held back for {treated[0]}: daily budget (2026-11-28)" in caplog.text
    tomorrow = engine.sweep(store, T0 + timedelta(days=1))
    assert [a["customer_id"] for a in tomorrow] == [treated[0]]


def test_a_sweeping_rule_with_bad_numbers_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = store_with(win_back="dry_run")
    quiet(store)
    path = "rules/win-back"
    store.documents[path] = {**store.documents[path], "condition": {"days": 7}}
    with caplog.at_level(logging.ERROR, logger="cfi_api.engine"):
        assert engine.sweep(store, T0) == []
    assert "Rule win-back was skipped" in caplog.text


# --- Rules that look back: Brand → Lovemark and Fad protection (T506) ---


def rescored(store: MemoryStore, order_id: str, quadrant: str, at: datetime) -> Scoring:
    """A new scoring of cust_1's order_id at time at, in quadrant, saved in
    their score history."""
    event_id = f"{order_id}-{at:%Y%m%d%H%M%S}"
    entry = {
        "event_id": event_id,
        "order_id": order_id,
        "at": at,
        "quadrant": quadrant,
        "contributions": {"fit": {"x3": 0.0}},
    }
    store.documents[f"customers/cust_1/scores/{event_id}"] = entry
    return Scoring(customer_id="cust_1", event_id=event_id, entry=entry)


def order(store: MemoryStore, quadrant: str) -> Scoring:
    """cust_1's next order, ord_1, ord_2, ..., a day after the last, scored
    once in quadrant: its scoring."""
    n = 1 + sum(path.startswith("orders/") for path in store.documents)
    order_id, placed_at = f"ord_{n}", T0 + timedelta(days=n)
    store.documents[f"orders/{order_id}"] = {
        "order_id": order_id,
        "customer_id": "cust_1",
        "placed_at": placed_at,
    }
    return rescored(store, order_id, quadrant, placed_at)


def decided_on(store: MemoryStore, *quadrants: str) -> list[str]:
    """The action ids decided as cust_1 places one order in each quadrant."""
    found = []
    for quadrant in quadrants:
        new = order(store, quadrant)
        found += [a["action_id"] for a in decide(store, new, new.entry["at"])]
    return found


def test_brand_to_lovemark_fires_after_three_brand_orders_in_a_row() -> None:
    store = store_with(brand_to_lovemark="dry_run")
    assert decided_on(store, "lovemark", "brand", "brand") == []
    third = order(store, "brand")
    decided = decide(store, third, third.entry["at"])
    assert [(a["action_id"], a["channel"], a["cost"]) for a in decided] == [
        ("brand-to-lovemark--ord_2", "in_app", 0)  # the streak's first order
    ]


def test_brand_to_lovemark_decides_once_per_streak() -> None:
    store = store_with(brand_to_lovemark="dry_run")
    first = ["brand"] * 5
    again = ["lovemark", "brand", "brand", "brand"]
    assert decided_on(store, *first, *again) == [
        "brand-to-lovemark--ord_1",
        "brand-to-lovemark--ord_7",
    ]


def test_brand_to_lovemark_counts_orders_as_they_were_placed() -> None:
    store = store_with(brand_to_lovemark="dry_run")
    assert decided_on(store, "brand", "lovemark", "brand", "brand") == []
    # ord_1 is rated after ord_4 was placed: it is still the oldest order.
    late = rescored(store, "ord_1", "brand", T0 + timedelta(days=5))
    assert decide(store, late, late.entry["at"]) == []


def test_brand_to_lovemarks_condition_sets_how_many_orders(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = store_with(brand_to_lovemark="dry_run")
    path = "rules/brand-to-lovemark"
    store.documents[path] = {**store.documents[path], "condition": {"orders": 2}}
    assert decided_on(store, "lovemark", "brand", "brand") == [
        "brand-to-lovemark--ord_2"
    ]
    for bad in ({"orders": 0}, {"orders": "3"}, {"orders": True}, {"days": 3}):
        store.documents[path] = {**store.documents[path], "condition": bad}
        with caplog.at_level(logging.ERROR, logger="cfi_api.engine"):
            assert decided_on(store, "brand") == []
    assert caplog.text.count("Rule brand-to-lovemark was skipped") == 4


@pytest.mark.parametrize("before", ["lovemark", "brand"])
def test_fad_protection_fires_when_the_newest_order_drops_into_fad(
    before: str,
) -> None:
    store = store_with(fad_protection="dry_run")
    assert decided_on(store, "product", before) == []
    dropped = order(store, "fad")
    decided = decide(store, dropped, dropped.entry["at"])
    assert [(a["action_id"], a["channel"], a["cost"]) for a in decided] == [
        ("fad-protection--ord_3", "in_app", 0)
    ]
    assert decide(store, dropped, dropped.entry["at"]) == []  # once per order


def test_fad_protection_needs_respect_to_have_fallen() -> None:
    store = store_with(fad_protection="dry_run")
    # The first order has nothing to fall from; then Product (Respect low
    # already) to Fad, and Fad to Fad.
    assert decided_on(store, "fad", "product", "fad", "fad") == []


def test_fad_protection_looks_only_at_the_newest_order() -> None:
    store = store_with(fad_protection="dry_run")
    assert decided_on(store, "lovemark", "lovemark") == []
    # ord_2 is rated after ord_3 was placed, and drops into Fad: too late.
    assert decided_on(store, "lovemark") == []
    late = rescored(store, "ord_2", "fad", T0 + timedelta(days=4))
    assert decide(store, late, late.entry["at"]) == []


def test_fad_protection_takes_no_numbers(caplog: pytest.LogCaptureFixture) -> None:
    store = store_with(fad_protection="dry_run")
    path = "rules/fad-protection"
    store.documents[path] = {**store.documents[path], "condition": {"drop": 0.1}}
    with caplog.at_level(logging.ERROR, logger="cfi_api.engine"):
        assert decided_on(store, "lovemark", "fad") == []
    assert caplog.text.count("Rule fad-protection was skipped") == 2


def test_history_is_read_only_when_a_rule_looks_back_and_once_per_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_with(
        fee_sensitivity="dry_run", brand_to_lovemark="dry_run", fad_protection="dry_run"
    )
    calls: list[str] = []
    for name in ("list_orders", "list_scores"):
        read = getattr(store, name)
        monkeypatch.setattr(
            store,
            name,
            lambda *a, _read=read, _name=name, **k: (
                calls.append(_name) or _read(*a, **k)
            ),
        )
    assert decided_on(store, "lovemark", "lovemark") == []
    assert calls == []  # neither Brand nor Fad: nothing to look back on
    assert decided_on(store, "brand") == []
    assert calls == ["list_orders", "list_scores"]  # one page each
