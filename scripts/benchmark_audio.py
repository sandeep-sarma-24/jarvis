#!/usr/bin/env python3
"""Phase 1 benchmark — measures actual VAD, ASR, and pipeline latencies.

Usage: python -m scripts.benchmark_audio
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import psutil  # type: ignore[import-untyped]

# Targets from architecture
TARGETS = {
    "vad_latency_ms": 1.0,
    "asr_latency_ms": 800.0,
    "e2e_latency_ms": 1000.0,
}


def _fmt(label: str, actual: float, target: float, unit: str = "ms") -> str:
    status = "PASS" if actual <= target else "FAIL"
    return f"  {label:30s}  target={target:>8.1f}{unit}  actual={actual:>8.1f}{unit}  [{status}]"


def run() -> None:
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / (1024 * 1024)
    print(f"RSS at start: {rss_start:.1f} MB")

    # ── VAD ──
    from app.audio.vad import VoiceActivityDetector

    t0 = time.perf_counter()
    vad = VoiceActivityDetector(threshold=0.5)
    vad.load()
    vad_load_ms = (time.perf_counter() - t0) * 1000
    print(f"\nVAD model load: {vad_load_ms:.0f} ms")

    chunk = np.zeros(512, dtype=np.float32)
    # warm up
    for _ in range(10):
        vad.process_chunk(chunk)
    vad.reset()

    latencies = []
    for _ in range(100):
        r = vad.process_chunk(chunk)
        latencies.append(r.latency_ms)
    vad.reset()

    vad_mean = sum(latencies) / len(latencies)
    vad_p99 = sorted(latencies)[98]
    print("\nVAD latency (100 chunks):")
    print(_fmt("mean", vad_mean, TARGETS["vad_latency_ms"]))
    print(_fmt("p99", vad_p99, TARGETS["vad_latency_ms"]))

    rss_after_vad = proc.memory_info().rss / (1024 * 1024)
    print(f"\nRSS after VAD: {rss_after_vad:.1f} MB")

    # ── ASR ──
    from app.audio.asr import SpeechRecognizer
    from app.config import ASRConfig

    t0 = time.perf_counter()
    asr = SpeechRecognizer(ASRConfig())
    asr.load()
    asr_load_ms = (time.perf_counter() - t0) * 1000
    print(f"\nASR model load: {asr_load_ms:.0f} ms")

    # ~2 seconds of white noise (simulates a speech segment)
    test_audio = np.random.randn(32000).astype(np.float32) * 0.01

    # warm up
    asr.transcribe(test_audio)

    asr_latencies = []
    for _ in range(5):
        r = asr.transcribe(test_audio)
        asr_latencies.append(r.latency_ms)

    asr_mean = sum(asr_latencies) / len(asr_latencies)
    print("\nASR latency (5 runs, 2s audio):")
    print(_fmt("mean", asr_mean, TARGETS["asr_latency_ms"]))

    rss_after_asr = proc.memory_info().rss / (1024 * 1024)
    print(f"\nRSS after ASR: {rss_after_asr:.1f} MB")

    # ── End-to-end ──
    from app.audio.segmentation import SpeechSegmenter, SpeechSegment
    from app.config import VADConfig

    segmenter = SpeechSegmenter(
        config=VADConfig(),
        sample_rate=16000,
        chunk_samples=512,
    )

    # simulate: 1s speech + silence end
    speech_audio = np.random.randn(512).astype(np.float32) * 0.01
    speech_chunks = 32  # ~1 second

    t0 = time.perf_counter()
    segment = None
    for _ in range(speech_chunks):
        from app.audio.vad import VADResult
        # simulate speech detection
        vr = VADResult(is_speech=True, confidence=0.9, latency_ms=0.1)
        segment = segmenter.process(speech_audio, vr)
    # end with silence
    for _ in range(15):
        vr = VADResult(is_speech=False, confidence=0.1, latency_ms=0.1)
        segment = segmenter.process(speech_audio, vr)
        if segment is not None:
            break

    if segment is not None:
        asr_r = asr.transcribe(segment.audio, segment.sample_rate)
        e2e = (time.perf_counter() - t0) * 1000
        print(f"\nEnd-to-end (segmentation + ASR):")
        print(_fmt("total", e2e, TARGETS["e2e_latency_ms"]))

    rss_end = proc.memory_info().rss / (1024 * 1024)
    print(f"\n── Memory summary ──")
    print(f"  RSS start:     {rss_start:.1f} MB")
    print(f"  RSS after VAD: {rss_after_vad:.1f} MB")
    print(f"  RSS after ASR: {rss_after_asr:.1f} MB")
    print(f"  RSS end:       {rss_end:.1f} MB")


if __name__ == "__main__":
    run()
