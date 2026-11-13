"""Simulated stability mode: no task, thread or memory growth across many turns."""
from __future__ import annotations

import asyncio

import pytest

from scripts import benchmark_e2e, stability_test
from scripts.e2e_fakes import FakePlayer, FakeStreamingModel, FakeTTS
from tests.test_scripts.test_benchmark_e2e import StubTurnManager


def test_simulated_stability_with_stub_turn_manager(monkeypatch) -> None:
    def build(cfg, args, bus, recorder):
        model = FakeStreamingModel(ttft_ms=2, tokens_per_second=2000)
        tts = FakeTTS(first_chunk_ms=2, rtf=0.05, seconds_per_char=0.0005, chunk_seconds=0.02)
        player = FakePlayer(max_queue_seconds=0.5)
        tm = StubTurnManager(model, tts, player, recorder)
        player.start(tts.sample_rate)
        return tm, player, [player.close, tts.close]

    monkeypatch.setattr(benchmark_e2e, "build_simulated", build)
    r = asyncio.run(stability_test.run_simulated(turns=30, batch=10))
    assert r["recorded"] == 40
    assert r["ok"], r


def test_simulated_stability_through_real_turn_manager() -> None:
    pytest.importorskip("app.orchestration.turns")
    r = asyncio.run(stability_test.run_simulated(turns=30, batch=10))
    assert r["ok"], r
