import pytest

# Every marker declared in pyproject.toml. Each needs a model, a remote host or
# real hardware, so a plain `pytest` skips it; select it with `-m <marker>`.
OPT_IN_MARKERS = ("integration", "hardware", "model")


def pytest_collection_modifyitems(config, items):
    """Auto-skip opt-in tests unless their marker appears in the -m expression."""
    expr = config.getoption("-m", default="") or ""
    for marker in OPT_IN_MARKERS:
        if marker in expr:
            continue
        skip = pytest.mark.skip(reason=f"needs -m {marker}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
