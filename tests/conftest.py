"""Shared fixtures for the sexpansion test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Environment variable pointing at the Java library's Output_examples folder.
JAVA_OUTPUTS_ENV = "SEXPANSION_JAVA_OUTPUTS"

_DEFAULT_JAVA_OUTPUTS = Path(__file__).resolve().parents[2] / "Sexpansion" / "Output_examples"


@pytest.fixture(scope="session")
def java_outputs() -> Path:
    """Directory with the captured stdout of the Java example programs.

    Tests using this fixture are skipped when the Java repository is not
    available (e.g. on CI).
    """
    path = Path(os.environ.get(JAVA_OUTPUTS_ENV, _DEFAULT_JAVA_OUTPUTS))
    if not path.is_dir():
        pytest.skip(f"Java baseline outputs not found at {path}")
    return path
