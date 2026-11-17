"""T309: the golden replay through fitcore, against notebooks/results."""

from pathlib import Path
from typing import Any

import pytest

from fitcore.golden import (
    differences,
    model_sha256,
    read_events,
    read_golden,
    replay,
)
from fitcore.model import ModelParams

RESULTS = Path(__file__).resolve().parents[2] / "notebooks/results"
MODEL = RESULTS / "t112_model.json"
EVENTS = RESULTS / "t309_events.jsonl"
GOLDEN = RESULTS / "t309_golden.json"

SCORES = {
    "respect": 0.25,
    "love": 0.75,
    "convenience": 0.5,
    "trust": 0.5,
    "fit": 0.5,
    "churn_risk": 0.1,
    "quadrant": "fad",
}


def result(**scores: Any) -> dict[str, Any]:
    return {"order_id": "ord_1", "scores": {**SCORES, **scores}}


def test_the_golden_file_was_made_with_this_model() -> None:
    assert read_golden(GOLDEN).model_sha256 == model_sha256(MODEL.read_bytes())


def test_fitcore_scores_the_log_as_the_golden_file_says() -> None:
    got = replay(ModelParams.from_json(MODEL.read_bytes()), read_events(EVENTS))
    assert differences(got, read_golden(GOLDEN).results) == []


def test_matching_results_have_no_differences() -> None:
    assert differences({"e1": result()}, {"e1": result()}) == []


@pytest.mark.parametrize(
    ("got", "line"),
    [
        (result(love=0.75 + 1e-9), "e1: love 0.750000001 for 0.75"),
        (result(quadrant="lovemark"), "e1: quadrant 'lovemark' for 'fad'"),
        ({"order_id": "ord_2", "scores": SCORES}, "e1: order ord_2 for ord_1"),
        ({"order_id": None, "scores": None}, "e1: order None for ord_1"),
    ],
)
def test_a_difference_is_reported(got: dict[str, Any], line: str) -> None:
    assert line in differences({"e1": got}, {"e1": result()})


def test_numbers_may_differ_by_the_tolerance_and_no_more() -> None:
    want = {"e1": result()}
    assert differences({"e1": result(love=0.75 + 5e-13)}, want) == []
    assert differences({"e1": result(love=0.75 + 5e-13)}, want, tolerance=0.0)


def test_a_missing_or_extra_event_is_reported() -> None:
    assert differences({}, {"e1": result()}) == ["e1: missing"]
    assert differences({"e1": result(), "e2": result()}, {"e1": result()}) == [
        "e2: not expected"
    ]
