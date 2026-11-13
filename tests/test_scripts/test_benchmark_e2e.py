"""Benchmark harness tests: table formatting, fakes, and the turn driver.

The driver is exercised against a small stand-in for TurnManager so the harness is
verified before app.orchestration.turns exists; the real one is tested at the end.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from argparse import Namespace
from functools import partial

import numpy as np
import pytest

from app.contracts import STAGES, Turn
from app.reasoning.model import StreamingReasoningModel
from app.tts.model import AudioChunk, StreamingAudioPlayer, StreamingTextToSpeech
from app.tts.player import PlaybackOutcome
from scripts import benchmark_e2e as bench
from scripts.e2e_fakes import CountingRecorder, FakePlayer, FakeStreamingModel, FakeTTS, make_transcript


def test_format_table_from_synthetic_recorder() -> None:
    rec = CountingRecorder()
    for i in range(100):
        turn = Turn(turn_id=str(i), speech_end_time=1000.0)
        turn.mark("asr_done", 1000.0 + (i + 1) / 1000)  # 1..100 ms
        rec.record(turn, "completed")
    table = bench.format_table("SIMULATED test", rec.summary(), {"extra": {"n": 0}})
    lines = table.splitlines()
    assert lines[0] == "SIMULATED test"
    asr = next(line for line in lines if line.startswith("asr_done")).split()
    assert asr == ["asr_done", "100", "50.0", "95.0", "99.0"]
    assert next(line for line in lines if line.startswith("first_audio")).split()[1:] == ["0", "-", "-", "-"]
    assert len(lines) == 2 + len(STAGES) - 1 + 1


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeStreamingModel(), StreamingReasoningModel)
    assert isinstance(FakeTTS(), StreamingTextToSpeech)
    assert isinstance(FakePlayer(), StreamingAudioPlayer)


def test_fake_model_streams_and_stops_on_cancel() -> None:
    model = FakeStreamingModel(ttft_ms=1, tokens_per_second=1000)
    assert "".join(model.stream([])).strip().startswith("Certainly.")
    cancel = threading.Event()
    cancel.set()
    assert list(model.stream([], cancel=cancel)) == []


def test_fake_player_plays_in_real_time_and_completes() -> None:
    done: list[tuple[str, PlaybackOutcome]] = []
    player = FakePlayer(on_done=lambda t, o: done.append((t, o)))
    player.start(1000)
    player.begin("a")
    t0 = time.time()
    assert player.feed(AudioChunk(np.zeros(100, np.float32), 1000), "a")  # 0.1 s
    player.end("a")
    deadline = time.time() + 2
    while not done and time.time() < deadline:
        time.sleep(0.005)
    player.close()
    assert done == [("a", PlaybackOutcome.COMPLETED)]
    assert player.last_block["a"] - t0 >= 0.08
    assert not player.feed(AudioChunk(np.zeros(10, np.float32), 1000), "b")  # not current


def test_fake_player_stop_interrupts_and_rejects_feed() -> None:
    player = FakePlayer(max_queue_seconds=0.2)
    player.start(1000)
    player.begin("a")
    player.feed(AudioChunk(np.zeros(150, np.float32), 1000), "a")
    blocked = []
    th = threading.Thread(target=lambda: blocked.append(player.feed(AudioChunk(np.zeros(150, np.float32), 1000), "a")))
    th.start()
    time.sleep(0.03)
    player.stop()
    th.join(1)
    player.close()
    assert blocked == [False]  # stop() wakes a feeder blocked on backpressure
    assert player.done["a"] is PlaybackOutcome.INTERRUPTED
    assert player.stale_played == 0
    assert player.stats["max_queue_seconds"] <= 0.2


class StubTurnManager:
    """Minimal contract-shaped TurnManager: latest wins, cancel via turn.cancel and player.stop()."""

    def __init__(self, model, tts, player, recorder) -> None:
        self.model, self.tts, self.player, self.recorder = model, tts, player, recorder
        self._turn: Turn | None = None
        self._task: asyncio.Task | None = None

    def on_first_audio(self, turn_id: str) -> None:
        if self._turn is not None and self._turn.turn_id == turn_id:
            self._turn.mark("first_audio")

    async def handle_transcript(self, transcript) -> None:
        self.on_barge_in(time.time())
        self._turn = Turn(uuid.uuid4().hex, transcript.speech_end_time)
        self._task = asyncio.create_task(self._run(self._turn))

    def on_barge_in(self, t: float) -> None:
        if self._turn is not None and not self._turn.cancelled:
            self._turn.cancel.set()
            self.player.stop()

    async def _run(self, turn: Turn) -> None:
        self.player.begin(turn.turn_id)
        text = "".join(await asyncio.to_thread(list, self.model.stream([], cancel=turn.cancel)))
        turn.mark("llm_done")
        for sentence in text.split(". "):
            if turn.cancelled:
                break
            await self.tts.stream_to(sentence, partial(self.player.feed, turn_id=turn.turn_id), cancel=turn.cancel)
        self.player.end(turn.turn_id)
        while turn.turn_id not in self.player.done:
            await asyncio.sleep(0.005)
        self.recorder.record(turn, "cancelled" if turn.cancelled else "completed")

    async def shutdown(self) -> None:
        self.on_barge_in(time.time())
        if self._task is not None:
            await self._task


def _args(**kw) -> Namespace:
    base = dict(turns=3, barge_in=False, barge_in_after_ms=50.0, ttft_ms=5.0, tps=500.0,
                tts_first_chunk_ms=5.0, tts_rtf=0.1, fail_rate=0.0, real=False, json=None)
    base.update(kw)
    return Namespace(**base)


async def _drive(args: Namespace):
    model = FakeStreamingModel(ttft_ms=args.ttft_ms, tokens_per_second=args.tps)
    tts = FakeTTS(first_chunk_ms=args.tts_first_chunk_ms, rtf=args.tts_rtf, seconds_per_char=0.002)
    player = FakePlayer(max_queue_seconds=0.5)
    recorder = CountingRecorder()
    tm = StubTurnManager(model, tts, player, recorder)
    bench._wire_player(tm, player)
    player.start(tts.sample_rate)

    async def transcripts():
        for i in range(args.turns):
            yield make_transcript("Jarvis, test", time.time())

    try:
        return await bench.drive_turns(tm, player, recorder, transcripts(), gap_s=0.0,
                                       barge_in_after_s=args.barge_in_after_ms / 1000 if args.barge_in else None)
    finally:
        await tm.shutdown()
        player.close()
        tts.close()


def test_driver_and_report_with_stub_turn_manager(capsys) -> None:
    args = _args()
    res = asyncio.run(_drive(args))
    out = bench.report(res, "SIMULATED", args)
    assert res.sent == res.recorder.count == 3 and res.timeouts == 0
    assert out["failure_rate"] == 0
    assert out["stages"]["all"]["first_audio"]["n"] == 3  # wired through on_first_audio
    assert out["stages"]["all"]["first_audio (player)"]["n"] == 3
    assert out["stale_chunks_played"] == 0
    assert "SIMULATED" in capsys.readouterr().out


def test_driver_barge_in_goes_silent_without_stale_audio() -> None:
    args = _args(barge_in=True, turns=2)
    res = asyncio.run(_drive(args))
    assert len(res.cancel_to_silence_ms) == 2
    assert max(res.cancel_to_silence_ms) < 100
    assert res.player.stale_played == 0
    assert {r["outcome"] for r in res.recorder.turns} == {"cancelled"}


def test_simulated_benchmark_through_real_turn_manager(tmp_path) -> None:
    pytest.importorskip("app.orchestration.turns")
    args = _args(json=str(tmp_path / "e2e.json"))
    res = asyncio.run(bench.run_simulated(args))
    out = bench.report(res, "SIMULATED", args)
    assert res.timeouts == 0
    assert out["stages"]["all"]["first_audio (player)"]["n"] == args.turns
    assert out["stale_chunks_played"] == 0


def test_main_reports_not_run_without_orchestration(capsys) -> None:
    try:
        import app.orchestration.turns  # noqa: F401
    except ImportError:
        assert bench.main([]) == 2
        assert "NOT RUN" in capsys.readouterr().out
    else:
        pytest.skip("orchestration present; covered by the simulated benchmark test")
