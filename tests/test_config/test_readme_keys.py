"""README.md must not drift from configs/default.toml: keys exist, stated values match."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()
with open(ROOT / "configs" / "default.toml", "rb") as f:
    TOML = tomllib.load(f)


def _lookup(path: str):
    node = TOML
    for part in path.split("."):
        node = node[part]  # KeyError -> key missing
    return node


def _readme_keys() -> set[str]:
    tokens = re.findall(r"`([a-z_]+(?:\.[a-z_]+)+)`", README)
    return {t for t in tokens if t.split(".")[0] in TOML}


def test_readme_mentions_config_keys() -> None:
    assert len(_readme_keys()) > 20


def test_every_readme_config_key_exists() -> None:
    missing = [k for k in sorted(_readme_keys()) if not _exists(k)]
    assert not missing, f"README mentions keys absent from configs/default.toml: {missing}"


def _exists(key: str) -> bool:
    try:
        _lookup(key)
        return True
    except (KeyError, TypeError):
        return False


def test_values_stated_in_readme_match_config() -> None:
    # Forms used in the README: `key` = 192 / false / 2.0 / `30m` / `""`
    pairs = re.findall(r"`([a-z_]+(?:\.[a-z_]+)+)` = (`[^`]*`|\w+(?:\.\d+)?)", README)
    assert pairs
    for key, stated in pairs:
        actual = _lookup(key)
        stated = stated.strip("`")
        if isinstance(actual, bool):
            assert stated == str(actual).lower(), key
        elif isinstance(actual, (int, float)):
            assert float(stated) == actual, key
        else:
            assert stated.strip('"') == actual, key
