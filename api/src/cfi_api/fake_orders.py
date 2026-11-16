"""Send simulated orders to a running API, to watch the matrix page move (T207).

From the repo root, with the API running (dashboard/README.md):

    python -m cfi_api.fake_orders

The events come from fitcore.sim_events, in time order, and are replayed at
--speed simulated days per real second: by default 25 customers' first 60
days, in about a minute. Sending the same run again changes nothing, as
every event comes back as already received, so a stopped run can be resumed
by starting it again. A new --seed adds new customers. Uses only the
standard library to talk HTTP.

An API with a write key (on Cloud Run, T205) needs it as --key, or in the
environment as CFI_WRITE_KEY. A dropped connection is retried, and so is an
event the API was too busy for (503, T302): every event has a fixed
event_id, so sending one again is always safe.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from fitcore.sim_events import simulate_events

from cfi_api.access import KEY_HEADER

# Sends one event and answers its HTTP status and body.
Send = Callable[[Mapping[str, Any]], tuple[int, str]]

DAY = timedelta(days=1)
REPORT_EVERY = 7  # simulated days between progress lines
RETRIES = 2  # more tries after a dropped connection or a busy answer
PAUSE = 0.5  # seconds before the first retry, more before later ones
BUSY = 503  # the API's answer when the customer has too many events at once


class Refused(Exception):
    """The API refused an event, so the run stopped there."""


def http_sender(
    url: str,
    key: str | None = None,
    *,
    timeout: float = 10.0,
    retries: int = RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Send:
    """A Send that POSTs each event to url's /api/v1/events.

    With a key, every request carries it as the X-CFI-Key header. A
    connection dropped before the answer (T205 saw 2 in about 90 requests
    through Hosting) is tried again up to retries times, PAUSE seconds apart
    and then longer; so is a 503, after the Retry-After seconds the API
    asks for. That is safe because of the event_id: if the API had already
    saved the event, it answers 200, already received. Raises
    ConnectionError when every try is dropped, and urllib.error.URLError
    when the API cannot be reached at all; a 503 on every try is answered.
    """
    endpoint = url.rstrip("/") + "/api/v1/events"
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers[KEY_HEADER] = key

    def send(event: Mapping[str, Any]) -> tuple[int, str]:
        data = json.dumps(event).encode()
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                endpoint, data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.status, response.read().decode()
            except urllib.error.HTTPError as err:
                if err.code == BUSY and attempt < retries:
                    pause = _retry_after(err) or PAUSE * (attempt + 1)
                    err.close()
                    print(
                        f"The API is busy ({err.code}); trying again in {pause:g} s.",
                        file=sys.stderr,
                    )
                    sleep(pause)
                    continue
                return err.code, err.read().decode(errors="replace")
            except OSError as err:
                if attempt == retries or not _dropped(err):
                    raise
                pause = PAUSE * (attempt + 1)
                print(
                    f"Connection dropped ({err}); trying again in {pause:g} s.",
                    file=sys.stderr,
                )
                sleep(pause)
        raise AssertionError("unreachable")  # the loop returns or raises

    return send


def _retry_after(err: urllib.error.HTTPError) -> float | None:
    """The seconds the answer's Retry-After header asks for, if it gives any."""
    try:
        return float(err.headers.get("Retry-After", ""))
    except ValueError:
        return None


def _dropped(err: OSError) -> bool:
    """Whether a connection broke after it was made: worth another try.

    A refused connection is not: nothing is listening, so main says so.
    """
    reason = err.reason if isinstance(err, urllib.error.URLError) else err
    return isinstance(reason, ConnectionError) and not isinstance(
        reason, ConnectionRefusedError
    )


def _when(event: Mapping[str, Any]) -> datetime:
    return datetime.fromisoformat(event["occurred_at"])


def replay(
    events: Sequence[Mapping[str, Any]],
    send: Send,
    speed: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] = print,
) -> Counter[int]:
    """Send events in order, each when its time comes, and count the answers.

    Event times are kept apart as they happened, at speed simulated days per
    real second; an event that is late goes at once. Every REPORT_EVERY
    simulated days, report says how far the run has got. Returns the count
    of each status (201 new, 200 already received). Raises Refused at the
    first other status, and ValueError if speed is not positive.
    """
    if not speed > 0:
        raise ValueError(f"speed must be positive, got {speed}")
    counts: Counter[int] = Counter()
    if not events:
        return counts
    first = _when(events[0])
    began = clock()
    reported = 0
    for event in events:
        day = (_when(event) - first) / DAY
        wait = day / speed - (clock() - began)
        if wait > 0:
            sleep(wait)
        if day >= reported + REPORT_EVERY:
            reported = int(day // REPORT_EVERY) * REPORT_EVERY
            report(f"day {reported}: {_summary(counts)}")
        status, body = send(event)
        if status not in (200, 201):
            raise Refused(
                f"{event['type']} {event['event_id']} got {status}: {body.strip()}"
            )
        counts[status] += 1
    return counts


def _summary(counts: Counter[int]) -> str:
    sent = counts.total()
    return f"{sent} events sent, {counts[201]} new, {counts[200]} already received"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfi_api.fake_orders",
        description="Send simulated orders to the API, in time order.",
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=25,
        help="simulated customers (default: %(default)s)",
    )
    parser.add_argument(
        "--days", type=float, default=60.0, help="simulated days (default: %(default)g)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="same seed, same events (default: %(default)s)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="simulated days per second (default: %(default)g)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="the API's address (default: %(default)s)",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("CFI_WRITE_KEY"),
        help="the API's write key, if it has one (default: $CFI_WRITE_KEY)",
    )
    args = parser.parse_args(argv)
    if args.customers < 1:
        parser.error("--customers must be at least 1")
    if not args.days > 0:
        parser.error("--days must be positive")
    if not args.speed > 0:
        parser.error("--speed must be positive")
    events = simulate_events(args.customers, args.seed, args.days)
    print(
        f"Sending {len(events)} events from {args.customers} customers' first "
        f"{args.days:g} days (seed {args.seed}) to {args.url}, "
        f"{args.speed:g} day(s) per second: about {args.days / args.speed:.0f} s."
    )
    try:
        counts = replay(events, http_sender(args.url, args.key), args.speed)
    except Refused as err:
        print(f"Stopped: {err}", file=sys.stderr)
        return 1
    except urllib.error.URLError as err:
        print(f"Cannot reach the API at {args.url} ({err.reason}).", file=sys.stderr)
        print("Is it running? See dashboard/README.md.", file=sys.stderr)
        return 1
    except ConnectionError as err:
        print(
            f"Lost the connection to {args.url} {RETRIES + 1} times in a row "
            f"({err}). Run it again to resume: events already sent change nothing.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Stopped. Run it again to resume: events already sent change nothing.")
        return 130
    print(f"Done: {_summary(counts)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
