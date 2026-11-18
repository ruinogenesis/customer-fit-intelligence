"""The model registry's admin commands (T310): python -m cfi_api.registry.

    python -m cfi_api.registry --project P register VERSION MODEL_JSON \\
        --card CARD --bucket BUCKET --trained YYYY-MM-DD --metric churn_auc=0.789
    python -m cfi_api.registry --project P list
    python -m cfi_api.registry --project P promote VERSION

register uploads a version's model.json and model card to
gs://BUCKET/models/VERSION/ and adds it as a candidate; promote makes it
live after checking that its file loads (cfi_api.model_registry). It runs
as you, with Application Default Credentials, as cfi_api.roles does.

This module is only the command line, so the API never imports it.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path

from cfi_api.model_registry import (
    Files,
    GcsFiles,
    Models,
    Registry,
    promote,
    register,
)


def _metric(text: str) -> tuple[str, float]:
    name, _, value = text.partition("=")
    if not name or not value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name, float(value)


def main(
    argv: Sequence[str] | None = None,
    *,
    store: Registry | None = None,
    files: Files | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfi_api.registry", description="The model registry (T310)."
    )
    parser.add_argument("--project", help="the Google Cloud project")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("register", help="upload a version as a candidate")
    add.add_argument("version")
    add.add_argument("model_json", type=Path)
    add.add_argument("--card", type=Path, required=True, help="its model_card.md")
    add.add_argument("--bucket", required=True, help="the models bucket")
    add.add_argument("--trained", type=date.fromisoformat, required=True)
    add.add_argument("--metric", type=_metric, action="append", default=[])
    commands.add_parser("list", help="every version, newest first")
    up = commands.add_parser("promote", help="make a version live")
    up.add_argument("version")
    args = parser.parse_args(argv)

    if store is None or files is None:
        if not args.project:
            print("Give --project, the project with the registry.", file=sys.stderr)
            return 2
        from google.cloud import firestore

        from cfi_api.firestore_store import FirestoreStore

        if store is None:
            store = FirestoreStore(firestore.Client(project=args.project))
        if files is None:
            files = GcsFiles(args.project)

    try:
        if args.command == "register":
            entry = register(
                store,
                files,
                version=args.version,
                model_json=args.model_json.read_bytes(),
                card=args.card.read_bytes(),
                bucket=args.bucket,
                trained=datetime.combine(args.trained, time(), UTC),
                metrics=dict(args.metric),
            )
            print(f"Registered {entry.version} as a candidate: {entry.path}")
        elif args.command == "promote":
            entry = promote(store, Models(files), args.version, datetime.now(UTC))
            print(f"{entry.version} is live.")
        for entry in store.list_models():
            print(f"  {entry.version:20} {entry.status:9} {entry.path}")
    except (ValueError, KeyError) as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
