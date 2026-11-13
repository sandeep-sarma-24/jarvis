"""configs/default.toml must match app/config.py: no ignored keys, documented values."""
from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from app.config import JarvisConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
TOML_PATH = ROOT / "configs" / "default.toml"


def _raw() -> dict:
    with open(TOML_PATH, "rb") as f:
        return tomllib.load(f)


def _toml_keys(table: dict, prefix: str = "") -> list[str]:
    """Dotted paths of every leaf key, e.g. 'perception.camera.enabled'."""
    out = []
    for k, v in table.items():
        path = f"{prefix}{k}"
        out.extend(_toml_keys(v, path + ".") if isinstance(v, dict) else [path])
    return out


def _field(path: str) -> dataclasses.Field:
    """Resolve a dotted key against the JarvisConfig dataclass tree, or raise KeyError."""
    obj: object = JarvisConfig()
    *parents, leaf = path.split(".")
    for part in parents:
        obj = getattr(obj, part)  # AttributeError -> unknown section
    fields = {f.name: f for f in dataclasses.fields(obj)}
    return fields[leaf]


def _value(cfg: JarvisConfig, path: str) -> object:
    obj: object = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("path", _toml_keys(_raw()))
def test_every_toml_key_maps_to_a_dataclass_field(path: str) -> None:
    # load_config drops unknown keys silently, so a typo here would be ignored at runtime
    try:
        _field(path)
    except (AttributeError, KeyError):
        pytest.fail(f"{path} in configs/default.toml has no field in app/config.py")


def test_every_toml_section_is_loaded() -> None:
    """A section with no matching JarvisConfig attribute would be skipped entirely."""
    names = {f.name for f in dataclasses.fields(JarvisConfig)}
    assert set(_raw()) <= names


def test_load_config_default_file_matches_toml_values() -> None:
    cfg = load_config(TOML_PATH)
    for path in _toml_keys(_raw()):
        expected = _raw()
        for part in path.split("."):
            expected = expected[part]
        assert _value(cfg, path) == expected, path


# Keys added for the streaming pipeline (docs/architecture/06-engineering-contracts.md)
# and for the local-only runtime.
DOCUMENTED = {
    "runtime.local_only": True,
    "reasoning.model": "gemma4:e2b-it-qat",
    "reasoning.ollama_url": "http://127.0.0.1:11434",
    "reasoning.num_ctx": 4096,
    "vad.speech_end_silence_ms": 192,
    "vad.pre_roll_ms": 96,
    "reasoning.connect_timeout_seconds": 2.0,
    "reasoning.keep_alive": "30m",
    "reasoning.think": False,
    "reasoning.fallback_url": "http://127.0.0.1:11434",
    "reasoning.fallback_model": "",
    "reasoning.fallback_cooldown_seconds": 300.0,
    "playback.max_queue_seconds": 2.0,
    "playback.tail_ms": 200,
    "barge_in.enabled": False,
    "barge_in.vad_threshold": 0.8,
    "barge_in.min_speech_ms": 256,
    "memory.cross_session": True,
    "perception.on_demand_timeout_seconds": 3.0,
    "perception.pause_during_turn": True,
}


@pytest.mark.parametrize("path,expected", sorted(DOCUMENTED.items()))
def test_load_config_yields_documented_values(path: str, expected: object) -> None:
    assert _value(load_config(TOML_PATH), path) == expected


# Where the dataclass default differs from the shipped file on purpose.
INTENTIONAL_DEFAULT_DIFFERENCES: set[str] = set()


@pytest.mark.parametrize("path", sorted(set(DOCUMENTED) - INTENTIONAL_DEFAULT_DIFFERENCES))
def test_new_key_dataclass_default_matches_toml(path: str) -> None:
    assert _value(JarvisConfig(), path) == _value(load_config(TOML_PATH), path)


def test_barge_in_disabled_by_default_in_code_and_file() -> None:
    assert JarvisConfig().barge_in.enabled is False
    assert load_config(TOML_PATH).barge_in.enabled is False


def test_missing_file_gives_dataclass_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "absent.toml") == JarvisConfig()
