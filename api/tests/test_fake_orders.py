"""T207: the fake-order script replays simulated events into the API."""

import json
import socket
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from fitcore.sim_events import simulate_events
from flask import Flask

from cfi_api import create_app, fake_orders
from cfi_api.fake_orders import Refused, Send, http_sender, replay
from cfi_api.store import MemoryStore

DAY0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def app(reference_model: str) -> Flask:
    return create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})


def client_sender(app: Flask) -> Send:
    """A Send that posts through Flask's test client instead of HTTP."""
    client = app.test_client()

    def send(event: Mapping[str, Any]) -> tuple[int, str]:
        response = client.post("/api/v1/events", json=event)
        return response.status_code, response.get_data(as_text=True)

    return send


class FakeTime:
    """A clock that moves only when the code under test sleeps, or a test
    moves it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def on_day(day: float) -> dict[str, Any]:
    """A stand-in event on a day of the run: replay reads only its time."""
    at = DAY0 + timedelta(days=day)
    return {
        "event_id": f"e{day}",
        "type": "order_placed",
        "occurred_at": at.isoformat(),
    }


def quiet(_: str) -> None:
    pass


def test_every_simulated_event_is_accepted_and_a_second_run_changes_nothing(
    app: Flask,
) -> None:
    events = simulate_events(5, seed=1, days=30.0)
    send = client_sender(app)
    time = FakeTime()
    first = replay(events, send, 1e6, clock=time.clock, sleep=time.sleep, report=quiet)
    assert first == {201: len(events)}
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    for customer_id in {event["customer_id"] for event in events}:
        customer = store.find_customer(customer_id)
        assert customer is not None and customer.scores is not None
    again = replay(events, send, 1e6, clock=time.clock, sleep=time.sleep, report=quiet)
    assert again == {200: len(events)}


def test_events_are_sent_as_far_apart_as_they_happened_at_speed() -> None:
    time = FakeTime()
    sent_at: list[float] = []

    def send(event: Mapping[str, Any]) -> tuple[int, str]:
        sent_at.append(time.now)
        return 201, "{}"

    events = [on_day(day) for day in (0.0, 1.0, 1.0, 3.5)]
    replay(events, send, 2.0, clock=time.clock, sleep=time.sleep, report=quiet)
    assert sent_at == pytest.approx([0.0, 0.5, 0.5, 1.75])


def test_a_late_event_goes_at_once() -> None:
    time = FakeTime()

    def slow(event: Mapping[str, Any]) -> tuple[int, str]:
        time.now += 1.0  # each send takes a second
        return 201, "{}"

    events = [on_day(day) for day in (0.0, 0.25, 0.5)]
    replay(events, slow, 1.0, clock=time.clock, sleep=time.sleep, report=quiet)
    assert time.sleeps == []


def test_progress_is_reported_every_simulated_week() -> None:
    time = FakeTime()
    lines: list[str] = []
    events = [on_day(day) for day in (0.0, 3.0, 7.5, 8.0, 15.0)]
    replay(
        events,
        lambda event: (201, "{}"),
        1e6,
        clock=time.clock,
        sleep=time.sleep,
        report=lines.append,
    )
    assert lines == [
        "day 7: 2 events sent, 2 new, 0 already received",
        "day 14: 4 events sent, 4 new, 0 already received",
    ]


def test_a_refused_event_stops_the_run() -> None:
    answers = iter([(201, "{}"), (409, '{"code": "conflict"}\n'), (201, "{}")])
    events = [on_day(day) for day in (0.0, 1.0, 2.0)]
    time = FakeTime()
    with pytest.raises(
        Refused, match='order_placed e1.0 got 409: {"code": "conflict"}$'
    ):
        replay(
            events,
            lambda event: next(answers),
            1e6,
            clock=time.clock,
            sleep=time.sleep,
            report=quiet,
        )
    assert next(answers) == (201, "{}")  # the third event was never sent


@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan")])
def test_speed_must_be_positive(speed: float) -> None:
    with pytest.raises(ValueError, match="speed must be positive"):
        replay([], lambda event: (201, "{}"), speed)


@pytest.fixture
def server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """A local HTTP server that answers 409 to order_issue and 201 to the rest."""
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(
                {"path": self.path, "type": self.headers["Content-Type"], "body": body}
            )
            refused = body["type"] == "order_issue"
            data = b'{"code": "conflict"}' if refused else b"{}"
            self.send_response(409 if refused else 201)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/", received
    httpd.shutdown()
    httpd.server_close()


def test_http_sender_posts_json_and_returns_errors_as_answers(
    server: tuple[str, list[dict[str, Any]]],
) -> None:
    url, received = server
    send = http_sender(url)
    assert send({"type": "order_placed"}) == (201, "{}")
    assert send({"type": "order_issue"}) == (409, '{"code": "conflict"}')
    assert [r["path"] for r in received] == ["/api/v1/events"] * 2
    assert {r["type"] for r in received} == {"application/json"}
    assert received[1]["body"] == {"type": "order_issue"}


def test_main_sends_the_run_and_says_how_it_went(
    app: Flask, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    urls: list[str] = []

    def sender(url: str, timeout: float = 10.0) -> Send:
        urls.append(url)
        return client_sender(app)

    monkeypatch.setattr(fake_orders, "http_sender", sender)
    argv = ["--customers", "3", "--days", "14", "--speed", "1e6", "--url", "http://x"]
    assert fake_orders.main(argv) == 0
    events = simulate_events(3, seed=1, days=14.0)
    out = capsys.readouterr().out
    assert urls == ["http://x"]
    assert out.startswith(f"Sending {len(events)} events from 3 customers' first 14")
    assert out.endswith(
        f"Done: {len(events)} events sent, {len(events)} new, 0 already received.\n"
    )


def test_main_says_when_the_api_cannot_be_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with socket.socket() as spare:
        spare.bind(("127.0.0.1", 0))
        port = spare.getsockname()[1]  # free once closed: nothing listens there
    url = f"http://127.0.0.1:{port}"
    argv = ["--customers", "3", "--days", "14", "--speed", "1e6", "--url", url]
    assert fake_orders.main(argv) == 1
    assert f"Cannot reach the API at {url}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--customers", "0"), ("--days", "0"), ("--speed", "0")],
)
def test_main_rejects_a_run_that_cannot_happen(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        fake_orders.main([flag, value])
    assert stopped.value.code == 2
