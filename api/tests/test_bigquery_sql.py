"""Every file in bigquery/ is SQL that BigQuery itself will accept, as far as
a test without BigQuery can tell.

sqlglot and the DuckDB stand-in accept a reserved keyword as a bare name,
but BigQuery refuses the whole file: T308's column `respect` (RESPECT, as in
RESPECT NULLS) ran in every test and failed on dev. Such a name needs
backticks.
"""

from pathlib import Path

import pytest
import sqlglot
from bigquery_on_duckdb import ROOT
from sqlglot import exp
from sqlglot.dialects.bigquery import BigQuery

FILES = sorted((ROOT / "bigquery").glob("*.sql"))


@pytest.mark.parametrize("path", FILES, ids=[path.name for path in FILES])
def test_no_reserved_keyword_is_a_bare_name(path: Path) -> None:
    reserved = BigQuery.Generator.RESERVED_KEYWORDS
    bare = {
        identifier.name
        for statement in sqlglot.parse(path.read_text(), read="bigquery")
        for identifier in statement.find_all(exp.Identifier)
        if not identifier.quoted and identifier.name.lower() in reserved
    }
    assert bare == set(), f"quote with backticks: {sorted(bare)}"


def test_every_file_is_checked() -> None:
    assert {path.name for path in FILES} >= {
        "raw.sql",
        "labels.sql",
        "features.sql",
        "features_daily.sql",
    }
