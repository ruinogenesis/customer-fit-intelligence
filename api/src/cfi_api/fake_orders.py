"""Send simulated orders to a running API, to watch the matrix page move (T207).

From the repo root, with the API running (dashboard/README.md):

    python -m cfi_api.fake_orders

The events come from fitcore.sim_events, in time order, and are replayed at
--speed simulated days per real second: by default 25 customers' first 60
days, in about a minute. Sending the same run again changes nothing, as
every event comes back as already received, so a stopped run can be resumed
by starting it again. A new --seed adds new customers. Uses only the
standard library to talk HTTP.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from fitcore.sim_events import simulate_events

# Sends one event and answers its HTTP status and body.
Send = Callable[[Mapping[str, Any]], tuple[int, str]]

DAY = timedelta(days=1)
REPORT_EVERY = 7  # simulated days between progress lines


class Refused(Exception):
    """The API refused an event, so the run stopped there."""


def http_sender(url: str, timeout: float = 10.0) -> Send:
    """A Send that POSTs each event to url's /api/v1/events.

    Raises urllib.error.URLError when the API cannot be reached.
    """
    endpoint = url.rstrip("/") + "/api/v1/events"

    def send(event: Mapping[str, Any]) -> tuple[int, str]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode(errors="replace")

    return send


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
        counts = replay(events, http_sender(args.url), args.speed)
    except Refused as err:
        print(f"Stopped: {err}", file=sys.stderr)
        return 1
    except urllib.error.URLError as err:
        print(f"Cannot reach the API at {args.url} ({err.reason}).", file=sys.stderr)
        print("Is it running? See dashboard/README.md.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped. Run it again to resume: events already sent change nothing.")
        return 130
    print(f"Done: {_summary(counts)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
