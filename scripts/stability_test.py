#!/usr/bin/env python3
"""Synthetic pipeline stability test.

Feeds simulated audio through VAD + segmentation + ASR + EventBus and measures
memory stability over time.  Does NOT use a real microphone.

This is NOT a substitute for a real microphone hardware test.

Usage: python -m scripts.stability_test [--minutes 10]
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import numpy as np
import psutil  # type: ignore[import-untyped]

from app.audio.vad import VoiceActivityDetector, VADResult
from app.audio.segmentation import SpeechSegmenter
from app.audio.asr import SpeechRecognizer
from app.config import ASRConfig, VADConfig
from app.events.bus import EventBus
from app.events.types import Modality, SpeechEvent

SR = 16000
CHUNK = 512


async def run(minutes: int = 10) -> None:
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
                        r = asr.transcribe(seg.audio, seg.sample_rate)
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
                        r = asr.transcribe(seg.audio, seg.sample_rate)
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


if __name__ == "__main__":
    minutes = 10
    if len(sys.argv) > 2 and sys.argv[1] == "--minutes":
        minutes = int(sys.argv[2])
    asyncio.run(run(minutes))
