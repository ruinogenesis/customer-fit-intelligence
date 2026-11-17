"""A DuckDB stand-in for BigQuery, so tests can run the SQL in bigquery/.

BigQuery has no emulator. sqlglot translates each view and table function
of a file to DuckDB, and raw.changes is filled with the rows the API's
export sends for a list of events, through the memory store (T306). The
tests check the SQL's logic; each task's dev check runs the same files on
BigQuery itself.
"""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb
import pytest
import sqlglot

from cfi_api import create_app
from cfi_api.export import MemoryPublisher
from cfi_api.store import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "notebooks/results/t112_model.json"
# raw.changes's columns that come from the message (bigquery/raw.sql).
COLUMNS = ("collection", "path", "event_id", "written_at", "document")
DATASETS = ("raw", "labels", "features")


def duckdb_sql(path: Path) -> list[str]:
    """The views and table functions in a BigQuery SQL file, as DuckDB SQL.

    A table function becomes a DuckDB table macro. Datasets and tables are
    left out: stand_in makes the datasets as DuckDB schemas, and the tests
    make the tables they need.
    """
    statements = []
    for statement in sqlglot.parse(path.read_text(), read="bigquery"):
        kind = statement.args.get("kind")
        if kind == "VIEW":
            statements.append(statement.sql(dialect="duckdb"))
        elif kind == "FUNCTION":
            function = statement.this
            name = function.this.sql(dialect="duckdb")
            params = ", ".join(param.name for param in function.expressions)
            body = statement.expression.sql(dialect="duckdb")
            statements.append(f"CREATE MACRO {name}({params}) AS TABLE {body}")
    return statements


def exported(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """raw.changes after events: one row per document the API saved."""
    with pytest.MonkeyPatch.context() as env:
        for name in [name for name in os.environ if name.startswith("CFI_")]:
            env.delenv(name)
        app = create_app({"MODEL_PATH": str(MODEL), "MODEL_VERSION": "t112-sim"})
    app.extensions["cfi_store"] = MemoryStore(outbox=True)
    publisher = app.extensions["cfi_publisher"] = MemoryPublisher()
    client = app.test_client()
    for event in events:
        assert client.post("/api/v1/events", json=event).status_code == 201
    return [json.loads(message) for message in publisher.sent]


def stand_in(rows: list[dict[str, Any]], *files: str) -> duckdb.DuckDBPyConnection:
    """DuckDB holding rows in raw.changes, with raw.sql's views and those of files.

    Times run in UTC, as BigQuery's TIMESTAMP does: DuckDB would otherwise
    use this computer's time zone when a timestamp loses its zone.
    """
    db = duckdb.connect()
    db.execute("SET TimeZone = 'UTC'")
    for dataset in DATASETS:
        db.execute(f"CREATE SCHEMA {dataset}")
    db.execute(
        "CREATE TABLE raw.changes (collection VARCHAR, path VARCHAR, "
        "event_id VARCHAR, written_at TIMESTAMPTZ, document JSON)"
    )
    # One column per parameter: far faster than a row at a time.
    db.execute(
        "INSERT INTO raw.changes SELECT "
        + ", ".join(f"unnest(${i + 1})" for i in range(len(COLUMNS))),
        [[row[column] for row in rows] for column in COLUMNS],
    )
    for name in ("raw.sql", *files):
        for statement in duckdb_sql(ROOT / "bigquery" / name):
            db.execute(statement)
    return db
