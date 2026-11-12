#!/usr/bin/env python3
"""End-to-end latency benchmark: end of speech to first audio, through TurnManager.

Default (SIMULATED): the real orchestration code (app.orchestration.turns.TurnManager,
attention, context builder, event bus) driven by the fakes in scripts/e2e_fakes.py.
No microphone, speaker, model or network. It measures orchestration overhead and
ordering, not model speed.

--real: Granite ASR, the configured reasoning backend and Phonon TTS. The spoken
input is synthesized by the TTS itself and played into a null sink, so no microphone
or speaker is used. The first turn is reported as cold, the rest as warm.

Usage:
  python -m scripts.benchmark_e2e [--turns 20] [--barge-in] [--json out.json]
  python -m scripts.benchmark_e2e --real [--turns 5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import resource
import sys
import time
from dataclasses import dataclass, field

from app.config import JarvisConfig, load_config
from app.contracts import STAGES, LatencyRecorder, percentiles

from scripts.e2e_fakes import (
    CountingRecorder,
    FakePlayer,
    FakeStreamingModel,
    FakeTTS,
    make_transcript,
)

PROMPTS = [
    "Jarvis, did the build pass?",
    "Jarvis, what time is the meeting?",
    "Jarvis, summarise the last error.",
    "Jarvis, how much disk space is left?",
]
TURN_TIMEOUT_S = 30.0


def format_table(title: str, summary: dict[str, dict[str, float]],
                 extra: dict[str, dict[str, float]] | None = None) -> str:
    """Render per-stage percentiles (ms since end of speech) as a fixed-width table."""
    lines = [title, f"{'stage':<26}{'n':>5}{'p50':>10}{'p95':>10}{'p99':>10}"]
    rows = {stage: summary.get(stage, {"n": 0}) for stage in STAGES[1:]}
    rows.update(extra or {})
    for name, p in rows.items():
        if not p.get("n"):
            lines.append(f"{name:<26}{0:>5}{'-':>10}{'-':>10}{'-':>10}")
        else:
            lines.append(f"{name:<26}{p['n']:>5}{p['p50']:>10.1f}{p['p95']:>10.1f}{p['p99']:>10.1f}")
    return "\n".join(lines)


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


@dataclass
class RunResult:
    recorder: CountingRecorder
    player: FakePlayer
    sent: int = 0
    timeouts: int = 0
    player_first_audio_ms: list[float] = field(default_factory=list)
    cancel_to_silence_ms: list[float] = field(default_factory=list)


def _attention_and_context(cfg: JarvisConfig):
    from app.attention.engine import AttentionEngine
    from app.context.builder import ContextBuilder
    from app.memory.working import WorkingMemory

    attention = AttentionEngine(
        assistant_names=cfg.attention.assistant_names,
        reason_threshold=cfg.attention.reason_threshold,
        working_memory=WorkingMemory(max_size=cfg.attention.working_memory_size),
    )
    context_builder = ContextBuilder(
        max_messages=cfg.reasoning.max_context_messages,
        max_characters=cfg.reasoning.max_context_characters,
        max_memory_items=cfg.memory.max_retrieved_memories,
        max_visual_events=cfg.perception.max_visual_events,
        max_visual_context_characters=cfg.perception.max_visual_context_characters,
    )
    return attention, context_builder


async def _until(pred, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.002)
    return True


async def drive_turns(tm, player: FakePlayer, recorder: CountingRecorder, transcripts, *,
                      barge_in_after_s: float | None = None, gap_s: float = 0.05) -> RunResult:
    """Send transcripts one at a time and wait for each turn to be recorded.

    `transcripts` yields (text, speech_end_time) lazily so speech_end is taken just
    before the turn starts. With barge_in_after_s, on_barge_in fires that long after
    the turn's first audible block.
    """
    res = RunResult(recorder=recorder, player=player)
    async for transcript in transcripts:
        res.sent += 1
        recorded, begun = recorder.count, len(player.begins)
        await tm.handle_transcript(transcript)
        if not await _until(lambda: len(player.begins) > begun or recorder.count > recorded, TURN_TIMEOUT_S):
            res.timeouts += 1
            continue
        tid = player.begins[-1] if len(player.begins) > begun else None
        if tid is not None and await _until(lambda: tid in player.first_audio or tid in player.done, TURN_TIMEOUT_S):
            if tid in player.first_audio:
                res.player_first_audio_ms.append((player.first_audio[tid] - transcript.speech_end_time) * 1000)
                if barge_in_after_s is not None:
                    await asyncio.sleep(barge_in_after_s)
                    cancel_t = time.time()
                    tm.on_barge_in(cancel_t)
                    await _until(lambda: tid in player.done, TURN_TIMEOUT_S)
                    silent_at = player.last_block.get(tid, cancel_t) + player.block_s
                    res.cancel_to_silence_ms.append(max(0.0, silent_at - cancel_t) * 1000)
        if not await _until(lambda: recorder.count > recorded, TURN_TIMEOUT_S):
            res.timeouts += 1
        await asyncio.sleep(gap_s)
    return res


def build_simulated(cfg: JarvisConfig, args, bus, recorder):
    """TurnManager wired to the fakes. Returns (turn_manager, player, closers)."""
    from app.orchestration.turns import TurnManager  # Agent E; absent until integration

    model = FakeStreamingModel(ttft_ms=args.ttft_ms, tokens_per_second=args.tps, fail_rate=args.fail_rate)
    tts = FakeTTS(first_chunk_ms=args.tts_first_chunk_ms, rtf=args.tts_rtf)
    player = FakePlayer(volume=cfg.playback.volume, max_queue_seconds=cfg.playback.max_queue_seconds)
    attention, context_builder = _attention_and_context(cfg)
    tm = TurnManager(cfg=cfg, bus=bus, attention=attention, context_builder=context_builder,
                     model=model, tts=tts, player=player, recorder=recorder)
    _wire_player(tm, player)
    player.start(tts.sample_rate)
    return tm, player, [player.close, tts.close, model.close]


def _wire_player(tm, player: FakePlayer) -> None:
    # The contract gives StreamingPlayer on_first_audio/on_done constructor callbacks but does
    # not name the TurnManager methods they go to; use them when TurnManager exposes them.
    for attr, names in (("on_first_audio", ("on_first_audio",)), ("on_done", ("on_playback_done", "on_done"))):
        for name in names:
            if callable(getattr(tm, name, None)):
                setattr(player, attr, getattr(tm, name))
                break


async def _transcripts(n: int):
    for i in range(n):
        yield make_transcript(PROMPTS[i % len(PROMPTS)], time.time())


async def run_simulated(args, cfg: JarvisConfig | None = None) -> RunResult:
    from app.events.bus import EventBus

    cfg = cfg or load_config()
    cfg.perception.enabled = False
    cfg.memory.enabled = False
    cfg.barge_in.enabled = bool(args.barge_in)
    recorder = CountingRecorder(maxlen=max(1000, args.turns))
    bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
    await bus.start()
    tm, player, closers = build_simulated(cfg, args, bus, recorder)
    try:
        return await drive_turns(tm, player, recorder, _transcripts(args.turns),
                                 barge_in_after_s=args.barge_in_after_ms / 1000 if args.barge_in else None)
    finally:
        await tm.shutdown()
        for close in closers:
            close()
        await bus.shutdown()


async def run_real(args) -> RunResult:
    """Real ASR, reasoning and TTS. Speech input is TTS output; audio goes to a null sink."""
    import numpy as np

    from app.audio.asr import SpeechRecognizer
    from app.contracts import Transcript
    from app.events.bus import EventBus
    from app.orchestration.turns import TurnManager
    from app.reasoning import build_reasoning_model
    from app.tts.local_tts import PhononTTS

    cfg = load_config()
    cfg.perception.enabled = False
    cfg.memory.enabled = False
    tts = PhononTTS(model_id=cfg.tts.model, voice=cfg.tts.voice, quant=cfg.tts.quant)
    asr = SpeechRecognizer(cfg.asr)
    await asyncio.to_thread(tts.load)
    await asyncio.to_thread(asr.load)
    model = build_reasoning_model(cfg.reasoning)  # no warm_up: the first turn is the cold one

    clips = []
    for text in PROMPTS:
        r = await asyncio.to_thread(tts.synthesize, text)
        if not r.success:
            raise SystemExit(f"TTS failed to synthesize benchmark input: {r.error}")
        n16 = int(len(r.audio) * 16000 / r.sample_rate)
        clips.append(np.interp(np.linspace(0, len(r.audio) - 1, n16), np.arange(len(r.audio)), r.audio)
                     .astype(np.float32))

    async def transcripts():
        for i in range(args.turns):
            audio = clips[i % len(clips)]
            t_end = time.time()  # the synthetic speaker has just stopped
            asr_result = await asr.transcribe_async(audio, 16000)
            yield Transcript(text=asr_result.text, speech_start_time=t_end - len(audio) / 16000,
                             speech_end_time=t_end, segment_emitted_time=t_end,
                             asr_latency_ms=asr_result.latency_ms, audio_seconds=len(audio) / 16000)

    recorder = CountingRecorder()
    bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
    await bus.start()
    player = FakePlayer(volume=cfg.playback.volume, max_queue_seconds=cfg.playback.max_queue_seconds)
    attention, context_builder = _attention_and_context(cfg)
    tm = TurnManager(cfg=cfg, bus=bus, attention=attention, context_builder=context_builder,
                     model=model, tts=tts, player=player, recorder=recorder)
    _wire_player(tm, player)
    player.start(tts.sample_rate)
    try:
        return await drive_turns(tm, player, recorder, transcripts(), gap_s=0.5)
    finally:
        await tm.shutdown()
        player.close()
        tts.close()
        asr.close()
        if hasattr(model, "close"):
            model.close()
        await bus.shutdown()


def report(res: RunResult, label: str, args) -> dict:
    rows = list(res.recorder.turns)
    outcomes = [r["outcome"] for r in rows]
    failures = outcomes.count("failed") + res.timeouts
    pfa = res.player_first_audio_ms
    groups = {"all": (rows, pfa)}
    if args.real:
        groups = {"cold (first turn)": (rows[:1], pfa[:1]), "warm": (rows[1:], pfa[1:])}
    out: dict = {"mode": label, "turns_sent": res.sent, "turns_recorded": res.recorder.count,
                 "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
                 "failure_rate": failures / res.sent if res.sent else None,
                 "timeouts": res.timeouts,
                 "max_player_queue_seconds": res.player.stats["max_queue_seconds"],
                 "stale_chunks_played": res.player.stale_played,
                 "peak_rss_mb": round(peak_rss_mb(), 1), "stages": {}}
    if not args.real:
        out["fake_timing"] = {"ttft_ms": args.ttft_ms, "tokens_per_second": args.tps,
                              "tts_first_chunk_ms": args.tts_first_chunk_ms, "tts_rtf": args.tts_rtf,
                              "fail_rate": args.fail_rate}
    for name, (group, first_audio) in groups.items():
        rec = LatencyRecorder(maxlen=max(1, len(group)))
        rec.turns.extend(group)
        summary = rec.summary()
        extra = {"first_audio (player)": percentiles(first_audio)}
        if args.barge_in:
            extra["cancel -> silence"] = percentiles(res.cancel_to_silence_ms)
        out["stages"][name] = {**summary, **extra}
        print(format_table(f"\n{label} — {name} — ms since end of speech", summary, extra))
    print(f"\nturns sent {res.sent}, recorded {res.recorder.count}, outcomes {out['outcomes']}, "
          f"timeouts {res.timeouts}")
    print(f"failure rate {out['failure_rate']}")
    print(f"max player queue {out['max_player_queue_seconds']:.2f} s, "
          f"stale chunks played {res.player.stale_played} (must be 0)")
    print(f"peak RSS {out['peak_rss_mb']} MB")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--turns", type=int, default=20)
    p.add_argument("--barge-in", action="store_true", help="barge in on every turn after first audio")
    p.add_argument("--barge-in-after-ms", type=float, default=300.0)
    p.add_argument("--ttft-ms", type=float, default=150.0, help="fake model time to first token")
    p.add_argument("--tps", type=float, default=40.0, help="fake model tokens per second")
    p.add_argument("--tts-first-chunk-ms", type=float, default=80.0)
    p.add_argument("--tts-rtf", type=float, default=0.2, help="fake TTS real-time factor")
    p.add_argument("--fail-rate", type=float, default=0.0, help="fraction of fake model calls that fail")
    p.add_argument("--real", action="store_true", help="real ASR, reasoning and TTS (loads models)")
    p.add_argument("--json", metavar="PATH", help="write results as JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.real and args.barge_in:
        raise SystemExit("--barge-in is simulated only")
    try:
        import app.orchestration.turns  # noqa: F401
    except ImportError:
        print("NOT RUN: app.orchestration.turns is not in this tree yet")
        return 2
    label = "REAL (null audio sink)" if args.real else "SIMULATED (fake models; orchestration overhead only)"
    res = asyncio.run(run_real(args) if args.real else run_simulated(args))
    out = report(res, label, args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
