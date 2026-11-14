"""Shared test setup: the reference model, and no CFI_* settings from the shell."""

import os
from pathlib import Path

import pytest

REFERENCE_MODEL = (
    Path(__file__).resolve().parents[2] / "notebooks/results/t112_model.json"
)


@pytest.fixture(autouse=True)
def no_cfi_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sets its own config, so CFI_* variables in your shell stay out."""
    for name in list(os.environ):
        if name.startswith("CFI_"):
            monkeypatch.delenv(name)


@pytest.fixture
def reference_model() -> str:
    """T112's model.json, the reference model for the API's tests."""
    return str(REFERENCE_MODEL)
