"""T207: the fake-order script replays simulated events into the API."""

import json
import socket
import threading
import urllib.error
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from fitcore.sim_events import simulate_events
from flask import Flask

from cfi_api import create_app, fake_orders
from cfi_api.access import KEY_HEADER
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
                {
                    "path": self.path,
                    "type": self.headers["Content-Type"],
                    "key": self.headers.get(KEY_HEADER),
                    "body": body,
                }
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
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
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
    assert {r["key"] for r in received} == {None}


def test_http_sender_sends_the_write_key_when_given_one(
    server: tuple[str, list[dict[str, Any]]],
) -> None:
    url, received = server
    assert http_sender(url, key="k3y")({"type": "order_placed"}) == (201, "{}")
    assert received[0]["key"] == "k3y"


@pytest.fixture
def dropping_server() -> Iterator[tuple[str, dict[str, int]]]:
    """A local server that drops the first state["drops"] connections without
    an answer, as Hosting sometimes did in T205, then answers the next
    state["busy"] requests 503 with Retry-After: 1, as the API does when a
    customer has too many events at once (T302), then answers 201."""
    state = {"drops": 0, "busy": 0, "seen": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            state["seen"] += 1
            if state["drops"] > 0:
                state["drops"] -= 1
                self.close_connection = True
                return  # no answer: the client sees RemoteDisconnected
            if state["busy"] > 0:
                state["busy"] -= 1
                self.send_response(503)
                self.send_header("Retry-After", "1")
                self.send_header("Content-Length", "6")
                self.end_headers()
                self.wfile.write(b'"busy"')
                return
            self.send_response(201)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", state
    httpd.shutdown()
    httpd.server_close()


def test_a_dropped_connection_is_tried_again(
    dropping_server: tuple[str, dict[str, int]], capsys: pytest.CaptureFixture[str]
) -> None:
    url, state = dropping_server
    state["drops"] = 2
    pauses: list[float] = []
    send = http_sender(url, sleep=pauses.append)
    assert send({"type": "order_placed"}) == (201, "{}")
    assert state["seen"] == 3
    assert pauses == [0.5, 1.0]
    err = capsys.readouterr().err
    assert err.count("Connection dropped") == 2
    assert "trying again in 0.5 s." in err
    assert "trying again in 1 s." in err


def test_three_dropped_connections_in_a_row_give_up(
    dropping_server: tuple[str, dict[str, int]],
) -> None:
    url, state = dropping_server
    state["drops"] = 3
    pauses: list[float] = []
    with pytest.raises(ConnectionError):
        http_sender(url, sleep=pauses.append)({"type": "order_placed"})
    assert state["seen"] == 3
    assert pauses == [0.5, 1.0]


def test_a_busy_answer_is_tried_again_after_retry_after(
    dropping_server: tuple[str, dict[str, int]], capsys: pytest.CaptureFixture[str]
) -> None:
    url, state = dropping_server
    state["busy"] = 2
    pauses: list[float] = []
    send = http_sender(url, sleep=pauses.append)
    assert send({"type": "order_placed"}) == (201, "{}")
    assert state["seen"] == 3
    assert pauses == [1.0, 1.0]
    assert capsys.readouterr().err.count("The API is busy (503)") == 2


def test_busy_on_every_try_is_answered(
    dropping_server: tuple[str, dict[str, int]],
) -> None:
    url, state = dropping_server
    state["busy"] = 3
    pauses: list[float] = []
    send = http_sender(url, sleep=pauses.append)
    assert send({"type": "order_placed"}) == (503, '"busy"')
    assert state["seen"] == 3
    assert pauses == [1.0, 1.0]


def test_a_refused_connection_is_not_tried_again() -> None:
    with socket.socket() as spare:
        spare.bind(("127.0.0.1", 0))
        port = spare.getsockname()[1]  # free once closed: nothing listens there
    pauses: list[float] = []
    with pytest.raises(urllib.error.URLError):
        http_sender(f"http://127.0.0.1:{port}", sleep=pauses.append)({})
    assert pauses == []


def test_main_says_when_the_connection_keeps_dropping(
    dropping_server: tuple[str, dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    url, state = dropping_server
    state["drops"] = 1000
    monkeypatch.setattr(fake_orders, "PAUSE", 0.0)
    argv = ["--customers", "3", "--days", "14", "--speed", "1e6", "--url", url]
    assert fake_orders.main(argv) == 1
    err = capsys.readouterr().err
    assert f"Lost the connection to {url} 3 times in a row" in err
    assert "Run it again to resume" in err


def test_main_sends_the_run_and_says_how_it_went(
    app: Flask, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    urls: list[str] = []
    keys: list[str | None] = []

    def sender(url: str, key: str | None = None, timeout: float = 10.0) -> Send:
        urls.append(url)
        keys.append(key)
        return client_sender(app)

    monkeypatch.setattr(fake_orders, "http_sender", sender)
    argv = ["--customers", "3", "--days", "14", "--speed", "1e6", "--url", "http://x"]
    assert fake_orders.main(argv) == 0
    events = simulate_events(3, seed=1, days=14.0)
    out = capsys.readouterr().out
    assert urls == ["http://x"]
    assert keys == [None]
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


@pytest.mark.parametrize(
    ("argv", "environ", "expected"),
    [
        (["--key", "from-flag"], {}, "from-flag"),
        ([], {"CFI_WRITE_KEY": "from-env"}, "from-env"),
        (["--key", "from-flag"], {"CFI_WRITE_KEY": "from-env"}, "from-flag"),
    ],
)
def test_main_passes_the_write_key_from_the_flag_or_the_environment(
    app: Flask,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    environ: dict[str, str],
    expected: str,
) -> None:
    keys: list[str | None] = []

    def sender(url: str, key: str | None = None, timeout: float = 10.0) -> Send:
        keys.append(key)
        return client_sender(app)

    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(fake_orders, "http_sender", sender)
    base = ["--customers", "1", "--days", "7", "--speed", "1e6"]
    assert fake_orders.main(base + argv) == 0
    assert keys == [expected]
