"""TTS benchmark: Phonon streaming synthesis with the real model and no audio output.

Run: python -m scripts.benchmark_tts [--runs 5]

Measures cold load time, first-chunk latency (cold first call, then warm) and
real-time factor (synthesis time / audio duration) for short, medium and long
sentences, as P50/P95/P99, plus process RSS before and after load.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from app.contracts import percentiles

SENTENCES = {
    "short": [
        "Done.",
        "Let me check.",
        "Sure, one moment.",
    ],
    "medium": [
        "According to the logs, the issue is probably caused by a timeout.",
        "I've found three relevant documents in your recent files.",
        "The build completed with two warnings and no errors.",
    ],
    "long": [
        "Based on the metrics from the last hour, memory usage has increased by fifteen "
        "percent, mostly in the indexing service, so a restart would likely help.",
        "The weather today is clear with temperatures around twenty degrees, rising to "
        "twenty five in the afternoon, and there is no rain expected until the weekend.",
    ],
}


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _fmt(stats: dict[str, float], unit: str) -> str:
    if not stats.get("n"):
        return "n=0"
    return (f"n={stats['n']} p50={stats['p50']:.3f}{unit} p95={stats['p95']:.3f}{unit} "
            f"p99={stats['p99']:.3f}{unit} min={stats['min']:.3f}{unit} max={stats['max']:.3f}{unit}")


async def _measure(tts, text: str) -> tuple[float | None, float]:
    """One streamed synthesis with audio discarded. Returns (first_chunk_ms, rtf)."""
    stats = await tts.stream_to(text, lambda chunk: True)
    rtf = (stats.synth_ms / 1000) / stats.audio_seconds if stats.audio_seconds else float("nan")
    return stats.first_chunk_ms, rtf


async def _run(runs: int) -> None:
    from app.config import load_config
    from app.tts.local_tts import PhononTTS

    cfg = load_config().tts
    tts = PhononTTS(model_id=cfg.model, voice=cfg.voice, quant=cfg.quant)

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    await asyncio.get_running_loop().run_in_executor(None, tts.load)
    load_ms = (time.perf_counter() - t0) * 1000
    rss_after = _rss_mb()

    print(f"Model: {tts.model_name}  quant={cfg.quant}  sample_rate={tts.sample_rate}")
    print(f"Cold load: {load_ms:.0f} ms")
    if rss_before is None:
        print("RSS: psutil not installed")
    else:
        print(f"RSS: before={rss_before:.0f} MB after load={rss_after:.0f} MB "
              f"delta={rss_after - rss_before:.0f} MB")

    cold_first, cold_rtf = await _measure(tts, SENTENCES["medium"][0])
    print(f"Cold first call: first_chunk={cold_first:.0f} ms rtf={cold_rtf:.3f}")

    all_first: list[float] = []
    for group, texts in SENTENCES.items():
        firsts: list[float] = []
        rtfs: list[float] = []
        for _ in range(runs):
            for text in texts:
                first, rtf = await _measure(tts, text)
                if first is not None:
                    firsts.append(first)
                rtfs.append(rtf)
        all_first += firsts
        print(f"[{group}] first_chunk_ms {_fmt(percentiles(firsts), '')}")
        print(f"[{group}] rtf           {_fmt(percentiles(rtfs), '')}")
    print(f"[all] warm first_chunk_ms {_fmt(percentiles(all_first), '')}")

    if rss_after is not None:
        print(f"RSS final: {_rss_mb():.0f} MB")
    tts.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5, help="repetitions per sentence")
    args = parser.parse_args()
    asyncio.run(_run(args.runs))


if __name__ == "__main__":
    main()
