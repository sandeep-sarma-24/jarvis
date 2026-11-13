#!/usr/bin/env python3
"""Synthetic pipeline stability test.

Default mode feeds simulated audio through VAD + segmentation + ASR + EventBus and
measures memory stability over time (loads the VAD and ASR models).

--simulated runs N turns, a third of them cancelled by barge-in, through
app.orchestration.turns.TurnManager with the fakes in scripts/e2e_fakes.py and
fails if live asyncio tasks or threads grow or RSS drifts. No models, no devices.

Neither mode uses a real microphone; this is NOT a substitute for a hardware test.

Usage:
  python -m scripts.stability_test [--minutes 10]
  python -m scripts.stability_test --simulated [--turns 300]
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
import threading
import time

import numpy as np
import psutil  # type: ignore[import-untyped]

from app.events.bus import EventBus

SR = 16000
CHUNK = 512


async def run(minutes: int = 10) -> None:
    from app.audio.asr import SpeechRecognizer
    from app.audio.segmentation import SpeechSegmenter
    from app.audio.vad import VADResult, VoiceActivityDetector
    from app.config import ASRConfig, VADConfig
    from app.events.types import Modality, SpeechEvent

    proc = psutil.Process(os.getpid())

    print(f"Synthetic pipeline stability test: {minutes} minutes")
    print(f"RSS at start: {proc.memory_info().rss / 1024 / 1024:.1f} MB")

    vad = VoiceActivityDetector(threshold=0.5)
    vad.load()

    asr = SpeechRecognizer(ASRConfig())
    asr.load()

    segmenter = SpeechSegmenter(config=VADConfig(), sample_rate=SR, chunk_samples=CHUNK)

    bus = EventBus(queue_size=256)
    event_count = 0

    async def counter(event):
        nonlocal event_count
        event_count += 1

    bus.subscribe(SpeechEvent, counter)
    await bus.start()

    rss_start = proc.memory_info().rss / 1024 / 1024
    print(f"RSS after model load: {rss_start:.1f} MB")

    chunks_processed = 0
    segments_count = 0
    transcripts_count = 0
    exceptions_count = 0
    start = time.time()
    half = minutes * 60 / 2

    silence = np.zeros(CHUNK, dtype=np.float32)
    noise = np.random.randn(CHUNK).astype(np.float32) * 0.01

    speech_interval = 30
    speech_chunks_count = int(1.5 * SR / CHUNK)
    silence_end_count = 15

    last_speech_time = -speech_interval
    rss_half = None

    while time.time() - start < minutes * 60:
        elapsed = time.time() - start

        if elapsed - last_speech_time >= speech_interval:
            for _ in range(speech_chunks_count):
                vad.process_chunk(silence)
                vr = VADResult(is_speech=True, confidence=0.9, latency_ms=0.1)
                seg = segmenter.process(noise, vr)
                chunks_processed += 1
                if seg is not None:
                    segments_count += 1
                    try:
                        r = await asr.transcribe_async(seg.audio, seg.sample_rate)
                        transcripts_count += 1
                        event = SpeechEvent(
                            session_id="stability",
                            source=Modality.AUDIO,
                            confidence=0.0,
                            transcript=r.text or "(empty)",
                            duration_seconds=seg.duration_seconds,
                        )
                        await bus.publish(event)
                    except Exception:
                        exceptions_count += 1

            for _ in range(silence_end_count):
                vr = VADResult(is_speech=False, confidence=0.1, latency_ms=0.1)
                seg = segmenter.process(silence, vr)
                chunks_processed += 1
                if seg is not None:
                    segments_count += 1
                    try:
                        r = await asr.transcribe_async(seg.audio, seg.sample_rate)
                        transcripts_count += 1
                        event = SpeechEvent(
                            session_id="stability",
                            source=Modality.AUDIO,
                            confidence=0.0,
                            transcript=r.text or "(empty)",
                            duration_seconds=seg.duration_seconds,
                        )
                        await bus.publish(event)
                    except Exception:
                        exceptions_count += 1

            last_speech_time = elapsed
        else:
            vr_real = vad.process_chunk(silence)
            segmenter.process(silence, vr_real)
            chunks_processed += 1

        if rss_half is None and elapsed >= half:
            rss_half = proc.memory_info().rss / 1024 / 1024
            print(f"\n[{elapsed:.0f}s] RSS at halfway: {rss_half:.1f} MB")
            print(f"  chunks={chunks_processed}, segments={segments_count}, transcripts={transcripts_count}")

        if chunks_processed % 1000 == 0:
            await asyncio.sleep(0)

    rss_end = proc.memory_info().rss / 1024 / 1024

    await bus.shutdown()
    vad.reset()

    print(f"\n── Synthetic Stability Test Results ({minutes} min) ──")
    print(f"  RSS start (after load): {rss_start:.1f} MB")
    if rss_half is not None:
        print(f"  RSS halfway:            {rss_half:.1f} MB")
    print(f"  RSS end:                {rss_end:.1f} MB")
    print(f"  RSS drift:              {rss_end - rss_start:+.1f} MB")
    print(f"  Chunks processed:       {chunks_processed}")
    print(f"  Segments detected:      {segments_count}")
    print(f"  Transcripts:            {transcripts_count}")
    print(f"  Events published:       {event_count}")
    print(f"  Exceptions:             {exceptions_count}")
    print(f"  Bus drops:              {bus.events_dropped}")

    drift = rss_end - rss_start
    stable = drift < 50
    print(f"\n  Memory stable: {'YES' if stable else 'NO — investigate leak'}")


RSS_DRIFT_LIMIT_MB = 30.0


async def run_simulated(turns: int = 300, batch: int = 15) -> dict[str, float]:
    """Simulated turns through TurnManager; returns growth figures and an `ok` flag."""
    from argparse import Namespace

    from app.config import load_config
    from scripts.benchmark_e2e import _transcripts, build_simulated, drive_turns
    from scripts.e2e_fakes import CountingRecorder

    proc = psutil.Process(os.getpid())
    cfg = load_config()
    cfg.perception.enabled = False
    cfg.memory.enabled = False
    cfg.barge_in.enabled = True
    fake = Namespace(ttft_ms=20.0, tps=400.0, tts_first_chunk_ms=10.0, tts_rtf=0.05, fail_rate=0.0)
    recorder = CountingRecorder(maxlen=100)
    bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
    await bus.start()
    tm, player, closers = build_simulated(cfg, fake, bus, recorder)

    async def batch_of(n: int, cancel: bool):
        return await drive_turns(tm, player, recorder, _transcripts(n), gap_s=0.0,
                                 barge_in_after_s=0.05 if cancel else None)

    def snapshot() -> tuple[int, int, float]:
        gc.collect()
        return len(asyncio.all_tasks()), threading.active_count(), proc.memory_info().rss / 2**20

    timeouts = 0
    try:
        timeouts += (await batch_of(batch, False)).timeouts  # warm-up: executors and caches fill here
        tasks0, threads0, rss0 = snapshot()
        done = 0
        while done < turns:
            n = min(batch, turns - done)
            timeouts += (await batch_of(n, cancel=(done // batch) % 3 == 2)).timeouts
            done += n
        tasks1, threads1, rss1 = snapshot()
    finally:
        await tm.shutdown()
        for close in closers:
            close()
        await bus.shutdown()

    result = {"turns": turns, "recorded": recorder.count, "timeouts": timeouts,
              "task_growth": tasks1 - tasks0, "thread_growth": threads1 - threads0,
              "rss_drift_mb": round(rss1 - rss0, 1), "stale_chunks_played": player.stale_played}
    result["ok"] = (result["task_growth"] <= 0 and result["thread_growth"] <= 0
                    and result["rss_drift_mb"] < RSS_DRIFT_LIMIT_MB and timeouts == 0
                    and player.stale_played == 0)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic stability test (no microphone).")
    parser.add_argument("--minutes", type=int, default=10, help="audio mode duration")
    parser.add_argument("--simulated", action="store_true", help="TurnManager with fakes, no models")
    parser.add_argument("--turns", type=int, default=300, help="simulated mode turn count")
    a = parser.parse_args()
    if not a.simulated:
        asyncio.run(run(a.minutes))
        sys.exit(0)
    try:
        import app.orchestration.turns  # noqa: F401
    except ImportError:
        print("NOT RUN: app.orchestration.turns is not in this tree yet")
        sys.exit(2)
    r = asyncio.run(run_simulated(a.turns))
    print("Simulated turn stability (SIMULATED, fake models):")
    for k, v in r.items():
        print(f"  {k:<20} {v}")
    sys.exit(0 if r["ok"] else 1)
