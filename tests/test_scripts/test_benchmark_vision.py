"""benchmark_vision reads the config and skips cleanly without a host (no network used)."""
import pytest

from app.config import load_config
from scripts import benchmark_vision


def test_model_and_url_come_from_config() -> None:
    vision = load_config().perception.vision
    assert (benchmark_vision.MODEL, benchmark_vision.OLLAMA_URL) == (vision.model, vision.ollama_url)


def test_stats_use_contract_percentiles() -> None:
    out = benchmark_vision._fmt_stats("x", [float(i) for i in range(1, 101)])
    assert "p50:      50.0 ms" in out and "p99:      99.0 ms" in out


def test_unreachable_host_prints_not_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(benchmark_vision, "_check_ollama_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        benchmark_vision.run()
    assert exc.value.code == 2
    assert "NOT RUN" in capsys.readouterr().out
