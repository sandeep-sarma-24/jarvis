"""TTS benchmark — measures Kokoro-82M synthesis performance.

Run: python -m scripts.benchmark_tts

Measures:
- Model initialization time
- Cold synthesis latency
- Warm synthesis latency (p50, p95)
- Audio duration
- Real-time factor (synthesis_time / audio_duration)
- Memory usage (RSS)
"""
from __future__ import annotations

import statistics
import time

import numpy as np


def _get_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", message=".*Language mismatch.*")
    warnings.filterwarnings("ignore", message=".*EspeakFallback.*")
    logging_module = __import__("logging")
    logging_module.getLogger("root").setLevel(logging_module.ERROR)

    print("=" * 60)
    print("JARVIS TTS Benchmark — Kokoro-82M via kokoro-mlx")
    print("=" * 60)

    rss_before = _get_rss_mb()

    # Model initialization
    print("\n[1] Model initialization...")
    t0 = time.perf_counter()
    from app.tts.local_tts import KokoroTTS
    tts = KokoroTTS()
    tts.load()
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    Init time: {init_ms:.0f}ms")

    rss_after_load = _get_rss_mb()
    print(f"    RSS after load: {rss_after_load:.1f}MB (delta: {rss_after_load - rss_before:.1f}MB)")

    # Cold synthesis
    print("\n[2] Cold synthesis (first call)...")
    test_text = "Hello, this is JARVIS. I'm ready to assist you."
    t0 = time.perf_counter()
    result = tts.synthesize(test_text)
    cold_ms = (time.perf_counter() - t0) * 1000
    assert result.success, f"Cold synthesis failed: {result.error}"
    print(f"    Latency: {cold_ms:.0f}ms")
    print(f"    Duration: {result.duration_seconds:.2f}s")
    print(f"    RTF: {cold_ms / (result.duration_seconds * 1000):.3f}x")

    # Warm synthesis — multiple iterations
    test_sentences = [
        "According to the logs, the issue is probably caused by a timeout.",
        "The weather today is clear with temperatures around twenty degrees.",
        "I've found three relevant documents in your recent files.",
        "That command has been executed successfully.",
        "Let me check the latest status for you.",
        "The build completed with two warnings and no errors.",
        "Based on the metrics, memory usage has increased by fifteen percent.",
        "I'll set a reminder for your meeting at three o'clock.",
    ]

    print(f"\n[3] Warm synthesis ({len(test_sentences)} sentences)...")
    latencies: list[float] = []
    durations: list[float] = []

    for sentence in test_sentences:
        t0 = time.perf_counter()
        result = tts.synthesize(sentence)
        lat = (time.perf_counter() - t0) * 1000
        assert result.success, f"Synthesis failed: {result.error}"
        latencies.append(lat)
        durations.append(result.duration_seconds)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    mean_lat = statistics.mean(latencies)
    mean_dur = statistics.mean(durations)
    mean_rtf = mean_lat / (mean_dur * 1000) if mean_dur > 0 else 0

    print(f"    Mean latency: {mean_lat:.0f}ms")
    print(f"    P50 latency:  {p50:.0f}ms")
    print(f"    P95 latency:  {p95:.0f}ms")
    print(f"    Min latency:  {min(latencies):.0f}ms")
    print(f"    Max latency:  {max(latencies):.0f}ms")
    print(f"    Mean audio duration: {mean_dur:.2f}s")
    print(f"    Mean RTF: {mean_rtf:.3f}x (< 1.0 = real-time)")

    # Streaming test
    print("\n[4] Streaming synthesis...")
    stream_text = "This is a streaming test. Multiple sentences allow chunked synthesis. Each chunk can start playing immediately."
    t0 = time.perf_counter()
    chunks = list(tts.synthesize_stream(stream_text))
    stream_ms = (time.perf_counter() - t0) * 1000
    total_samples = sum(len(c.audio) for c in chunks)
    stream_duration = total_samples / 24000

    first_chunk_samples = len(chunks[0].audio) if chunks else 0
    print(f"    Total chunks: {len(chunks)}")
    print(f"    Total latency: {stream_ms:.0f}ms")
    print(f"    Total duration: {stream_duration:.2f}s")
    print(f"    First chunk samples: {first_chunk_samples}")

    # Memory
    rss_final = _get_rss_mb()
    print(f"\n[5] Memory")
    print(f"    RSS before: {rss_before:.1f}MB")
    print(f"    RSS after load: {rss_after_load:.1f}MB")
    print(f"    RSS final: {rss_final:.1f}MB")
    print(f"    TTS model delta: {rss_after_load - rss_before:.1f}MB")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model:              Kokoro-82M (bf_emma)")
    print(f"  Sample rate:        24000 Hz")
    print(f"  Init time:          {init_ms:.0f}ms")
    print(f"  Cold latency:       {cold_ms:.0f}ms")
    print(f"  Warm P50:           {p50:.0f}ms")
    print(f"  Warm P95:           {p95:.0f}ms")
    print(f"  Mean RTF:           {mean_rtf:.3f}x")
    print(f"  Memory (model):     {rss_after_load - rss_before:.1f}MB")
    print(f"  Streaming chunks:   {len(chunks)}")
    print("=" * 60)

    tts.close()


if __name__ == "__main__":
    main()
