"""Every file in bigquery/, and every query the API sends to BigQuery, is SQL
that BigQuery itself will accept, as far as a test without BigQuery can tell.

sqlglot and the DuckDB stand-in accept a reserved keyword as a bare name,
but BigQuery refuses the whole file: T308's column `respect` (RESPECT, as in
RESPECT NULLS) ran in every test and failed on dev. Such a name needs
backticks.
"""

import pytest
import sqlglot
from bigquery_on_duckdb import ROOT
from sqlglot import exp
from sqlglot.dialects.bigquery import BigQuery

from cfi_api import history

FILES = {path.name: path.read_text() for path in sorted(ROOT.glob("bigquery/*.sql"))}
# The SQL in cfi_api (T313), by the name it has there.
QUERIES = {
    "cfi_api.history.MOVEMENT": history.MOVEMENT,
    "cfi_api.history.DRIVERS": history.DRIVERS,
}
SOURCES = FILES | QUERIES


@pytest.mark.parametrize("sql", SOURCES.values(), ids=SOURCES.keys())
def test_no_reserved_keyword_is_a_bare_name(sql: str) -> None:
    reserved = BigQuery.Generator.RESERVED_KEYWORDS
    bare = {
        identifier.name
        for statement in sqlglot.parse(sql, read="bigquery")
        for identifier in statement.find_all(exp.Identifier)
        if not identifier.quoted and identifier.name.lower() in reserved
    }
    assert bare == set(), f"quote with backticks: {sorted(bare)}"


def test_every_file_is_checked() -> None:
    assert set(FILES) >= {
        "raw.sql",
        "labels.sql",
        "features.sql",
        "features_daily.sql",
    }
