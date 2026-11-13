import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


def test_trusted_returns_bool():
    pytest.importorskip("ApplicationServices")
    from app.screen.ax import AXBackend
    assert isinstance(AXBackend().trusted(), bool)
