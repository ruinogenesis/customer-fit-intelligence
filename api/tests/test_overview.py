"""T402: GET /overview and GET /overview/trend, the Overview screen's numbers.

/overview is checked against the customers GET /customers lists, on the
memory store and, under the emulator, on Firestore (whose avg() and count()
must agree with Python's). /overview/trend runs its SQL on the DuckDB
stand-in for BigQuery over the golden log, as test_movement does, and is
compared with the same daily rows read in Python.
"""

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import duckdb
import pytest
from bigquery_on_duckdb import MODEL
from fitcore.model import QUADRANTS
from flask.testing import FlaskClient
from test_access import bearer
from test_customers import OURS, _app_with_events, everything, needs_emulator
from test_features import FIRST, LAST
from test_movement import _app, _today, db, rows, snapshot  # noqa: F401 (fixtures)

from cfi_api import create_app
from cfi_api.access import ACCESS, EMULATOR
from cfi_api.overview import AT_RISK, DAYS

API = "/api/v1"
FIT, CHURN, QUADRANT = 9, 10, 11  # a snapshot row's (test_features.expected)
CLOSE = 1e-12  # Firestore's avg() sums in its own order


@pytest.fixture(
    scope="module", params=["memory", pytest.param("firestore", marks=needs_emulator)]
)
def client(request: pytest.FixtureRequest) -> FlaskClient:
    """An app on each store, after test_customers' events."""
    return _app_with_events(request.param)


def overview(client: FlaskClient, **query: Any) -> Any:
    answer = client.get(f"{API}/overview", query_string=query)
    assert answer.status_code == 200, answer.get_json()
    return answer.get_json()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# --- /overview: now ---


@pytest.mark.parametrize("at_risk", [0.0, 0.02, AT_RISK, 1.0])
def test_the_overview_sums_up_the_listed_customers(
    client: FlaskClient, at_risk: float
) -> None:
    listed = everything(client, f"{API}/customers", "customers", limit=200)
    scored = [c for c in listed if c["scores"] is not None]
    got = overview(client, at_risk=at_risk)
    assert got["customers"] == len(listed)
    assert got["unscored"] == len(listed) - len(scored)
    for quadrant in QUADRANTS:
        assert got["quadrants"][quadrant] == sum(
            c["quadrant"] == quadrant for c in listed
        )
    assert got["average_fit"] == pytest.approx(
        _mean([c["scores"]["fit"] for c in scored]), abs=CLOSE
    )
    assert got["average_churn_risk"] == pytest.approx(
        _mean([c["churn_risk"] for c in scored]), abs=CLOSE
    )
    risky = sum(c["churn_risk"] >= at_risk for c in scored)
    assert got["at_risk"] == {"threshold": at_risk, "customers": risky}


def test_the_overview_counts_this_files_customers() -> None:
    got = overview(_app_with_events("memory"))
    assert got["customers"] == len(OURS)
    assert got["unscored"] == 1
    assert all(got["quadrants"][quadrant] > 0 for quadrant in QUADRANTS)
    assert got["at_risk"]["threshold"] == AT_RISK
    assert 0 < got["average_fit"] < 1 and 0 < got["average_churn_risk"] < 1


def test_before_anyone_is_scored_the_averages_are_null() -> None:
    got = overview(_app(None).test_client())
    assert got == {
        "customers": 0,
        "unscored": 0,
        "quadrants": dict.fromkeys(QUADRANTS, 0),
        "average_fit": None,
        "average_churn_risk": None,
        "at_risk": {"threshold": AT_RISK, "customers": 0},
    }


@pytest.mark.parametrize(
    "query", ["at_risk=1.5", "at_risk=-0.1", "at_risk=x", "risk=0.1"]
)
def test_a_bad_overview_query_is_422(query: str) -> None:
    answer = _app(None).test_client().get(f"{API}/overview?{query}")
    assert answer.status_code == 422


# --- /overview/trend: each day, from BigQuery ---


@pytest.fixture(scope="module")
def trending(db: duckdb.DuckDBPyConnection) -> FlaskClient:  # noqa: F811
    """An app whose history is the stand-in."""
    return _app(db).test_client()


def trend(client: FlaskClient, **query: Any) -> Any:
    answer = client.get(f"{API}/overview/trend", query_string=query)
    assert answer.status_code == 200, answer.get_json()
    return answer.get_json()


def want_days(
    snapshot: dict[tuple[date, str], tuple[Any, ...]],  # noqa: F811
    first: date,
    last: date,
) -> list[dict[str, Any]]:
    """Each day's numbers, read from the Python snapshot."""
    by_day: dict[date, list[tuple[Any, ...]]] = defaultdict(list)
    for (day, _), row in snapshot.items():
        if first <= day <= last:
            by_day[day].append(row)
    return [
        {
            "day": str(day),
            "customers": len(held),
            "average_fit": sum(r[FIT] for r in held) / len(held),
            "average_churn_risk": sum(r[CHURN] for r in held) / len(held),
            "quadrants": {q: sum(r[QUADRANT] == q for r in held) for q in QUADRANTS},
        }
        for day, held in sorted(by_day.items())
    ]


def same_days(got: list[dict[str, Any]], want: list[dict[str, Any]]) -> None:
    assert [d["day"] for d in got] == [d["day"] for d in want]
    for g, w in zip(got, want, strict=True):
        assert (g["customers"], g["quadrants"]) == (w["customers"], w["quadrants"])
        assert g["average_fit"] == pytest.approx(w["average_fit"], abs=CLOSE)
        assert g["average_churn_risk"] == pytest.approx(
            w["average_churn_risk"], abs=CLOSE
        )


@pytest.mark.parametrize("days", [1, 7, DAYS, 61])
def test_the_trend_is_each_days_averages_and_counts(
    trending: FlaskClient,
    snapshot: dict[tuple[date, str], tuple[Any, ...]],  # noqa: F811
    days: int,
) -> None:
    got = trend(trending, days=days, to=LAST)
    first = LAST - timedelta(days=days - 1)
    assert (got["from"], got["to"]) == (str(first), str(LAST))
    assert len(got["days"]) == days  # the golden log scores someone every day
    same_days(got["days"], want_days(snapshot, first, LAST))


def test_days_before_the_first_scoring_are_left_out(
    trending: FlaskClient,
    snapshot: dict[tuple[date, str], tuple[Any, ...]],  # noqa: F811
) -> None:
    last = FIRST + timedelta(days=2)
    got = trend(trending, days=10, to=last)
    assert got["from"] == str(last - timedelta(days=9))
    same_days(got["days"], want_days(snapshot, FIRST, last))
    assert len(got["days"]) == 3


def test_by_default_the_trend_is_four_weeks_to_yesterday(trending: FlaskClient) -> None:
    yesterday = _today() - timedelta(days=1)
    got = trend(trending)
    assert (got["from"], got["to"]) == (
        str(yesterday - timedelta(days=DAYS - 1)),
        str(yesterday),
    )
    assert got["days"] == []  # the golden log ends in March


BAD_TRENDS = {
    "no days": "days=0",
    "too many days": "days=91",
    "today": f"to={_today()}",
    "future": f"to={_today() + timedelta(days=3)}",
    "not a date": "to=yesterday",
    "unknown": "since=2026-01-01",
}


@pytest.mark.parametrize("query", BAD_TRENDS.values(), ids=BAD_TRENDS.keys())
def test_a_bad_trend_query_is_422(trending: FlaskClient, query: str) -> None:
    assert trending.get(f"{API}/overview/trend?{query}").status_code == 422


def test_without_bigquery_the_trend_is_503() -> None:
    answer = _app(None).test_client().get(f"{API}/overview/trend")
    assert answer.status_code == 503
    assert "no trend" in answer.get_json()["message"]


# --- Who may, and the docs ---


def test_every_staff_role_reads_the_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EMULATOR, "127.0.0.1:9099")
    client = _app_with_auth()
    for path in ("/overview", "/overview/trend"):
        assert client.get(f"{API}{path}").status_code == 401
    assert client.get(f"{API}/overview", headers=bearer("viewer")).status_code == 200
    assert ACCESS["overview.Overview", "GET"].role == "viewer"
    assert ACCESS["overview.Trend", "GET"].role == "viewer"


def _app_with_auth() -> FlaskClient:
    with pytest.MonkeyPatch.context() as env:
        for name in ("CFI_HISTORY", "CFI_STORE"):
            env.delenv(name, raising=False)
        app = create_app(
            {
                "MODEL_PATH": str(MODEL),
                "MODEL_VERSION": "t112-sim",
                "AUTH": "firebase",
                "PROJECT": "demo-cfi",
            }
        )
    return app.test_client()


def test_the_overview_endpoints_are_documented(trending: FlaskClient) -> None:
    paths = trending.get(f"{API}/openapi.json").get_json()["paths"]
    now = paths[f"{API}/overview"]["get"]
    daily = paths[f"{API}/overview/trend"]["get"]
    assert {"200", "401", "403", "422"} <= set(now["responses"])
    assert {"200", "401", "403", "422", "503"} <= set(daily["responses"])
    assert {p["name"] for p in now["parameters"]} == {"at_risk"}
    assert {p["name"] for p in daily["parameters"]} == {"days", "to"}
