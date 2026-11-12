"""Every pytest marker declared in pyproject.toml is skipped by default in conftest."""
import tomllib
from pathlib import Path

from tests.conftest import OPT_IN_MARKERS

ROOT = Path(__file__).resolve().parents[2]


def test_every_declared_marker_is_opt_in() -> None:
    with open(ROOT / "pyproject.toml", "rb") as f:
        declared = tomllib.load(f)["tool"]["pytest"]["ini_options"]["markers"]
    assert {m.split(":")[0].strip() for m in declared} == set(OPT_IN_MARKERS)
