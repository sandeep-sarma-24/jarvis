import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration/hardware tests unless explicitly selected."""
    run_integration = config.getoption("-m", default="") and "integration" in config.getoption("-m", default="")
    run_hardware = config.getoption("-m", default="") and "hardware" in config.getoption("-m", default="")

    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(pytest.mark.skip(reason="needs -m integration"))
        if "hardware" in item.keywords and not run_hardware:
            item.add_marker(pytest.mark.skip(reason="needs -m hardware"))
