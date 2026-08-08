#!/usr/bin/env python3
"""Phase 2 benchmark — measures attention evaluation latency.

Usage: python -m scripts.benchmark_attention
"""
from __future__ import annotations

import statistics
import time

from app.attention.engine import AttentionEngine
from app.events.types import Modality, SpeechEvent

CASES = [
    "Hey Jarvis, what time is it?",
    "Why is the API failing?",
    "I'll call you tomorrow.",
    "That's interesting.",
    "",
    "Explain this error message to me please.",
    "Jarvis is a fictional character.",
    "The server is throwing a 500 error on the auth endpoint.",
    "Can you help me debug this function?",
    "Yeah.",
]


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(session_id="bench", source=Modality.AUDIO,
                       transcript=text, duration_seconds=1.0)


def run() -> None:
    engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)

    # Warm up
    for text in CASES:
        engine.evaluate(_speech(text))

    iterations = 1000
    all_times: list[float] = []
    rule_times: list[float] = []
    scorer_times: list[float] = []

    for _ in range(iterations):
        for text in CASES:
            t0 = time.perf_counter_ns()
            d = engine.evaluate(_speech(text))
            elapsed_ns = time.perf_counter_ns() - t0
            elapsed_ms = elapsed_ns / 1_000_000

            all_times.append(elapsed_ms)
            if d.tier == "rule":
                rule_times.append(elapsed_ms)
            elif d.tier == "scorer":
                scorer_times.append(elapsed_ms)

    def _report(label: str, times: list[float], target_ms: float) -> None:
        if not times:
            print(f"  {label}: no samples")
            return
        times_sorted = sorted(times)
        n = len(times_sorted)
        mean = statistics.mean(times_sorted)
        p50 = times_sorted[n // 2]
        p95 = times_sorted[int(n * 0.95)]
        p99 = times_sorted[int(n * 0.99)]
        status = "PASS" if p99 <= target_ms else "MISS"
        print(f"  {label}:")
        print(f"    samples: {n}")
        print(f"    mean:    {mean:.4f} ms")
        print(f"    p50:     {p50:.4f} ms")
        print(f"    p95:     {p95:.4f} ms")
        print(f"    p99:     {p99:.4f} ms  (target: {target_ms} ms) [{status}]")

    print(f"\n── Attention Benchmark ({iterations} iterations × {len(CASES)} cases) ──\n")
    _report("Total (all tiers)", all_times, 10.0)
    print()
    _report("Tier 1 (rules)", rule_times, 0.1)
    print()
    _report("Tier 2 (scorer)", scorer_times, 5.0)

    print(f"\n  Counters: {engine.counters}")


if __name__ == "__main__":
    run()
