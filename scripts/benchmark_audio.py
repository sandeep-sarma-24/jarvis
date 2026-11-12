#!/usr/bin/env python3
"""Audio benchmark: VAD per-chunk cost, ASR latency / real-time factor, endpointing delay.

Usage:
    python -m scripts.benchmark_audio [--wav speech.wav] [--asr-runs 10] [--vad-chunks 2000]
    python -m scripts.benchmark_audio --skip-models   # endpointing only, no model loads

No microphone is opened. Without --wav, ASR runs on synthetic audio (latency
and RTF are meaningful; the transcript is not).
"""
from __future__ import annotations

import argparse
import asyncio
import time
import wave

import numpy as np

from app.config import load_config
from app.contracts import percentiles

SR = 16000


def _print(label: str, stats: dict[str, float], unit: str = "ms") -> None:
    if not stats.get("n"):
        print(f"  {label:28s} n=0")
        return
    cols = "  ".join(f"{k}={stats[k]:.3f}" for k in ("p50", "p95", "p99", "max"))
    print(f"  {label:28s} n={stats['n']:<5d} {cols} {unit}")


def load_wav(path: str) -> np.ndarray:
    """16-bit PCM WAV → float32 mono at 16 kHz (linear resample if needed)."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("--wav must be 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = data.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768.0
    if rate != SR:
        audio = np.interp(np.arange(0, len(audio), rate / SR), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def synthetic_speech(seconds: float, seed: int = 0) -> np.ndarray:
    """Voice-like signal: harmonic stack with a syllable-rate envelope plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 8))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    return (0.2 * voiced * envelope + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def bench_vad(audio: np.ndarray, n_chunks: int, threshold: float) -> None:
    from app.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(threshold=threshold)
    t0 = time.perf_counter()
    vad.load()
    print(f"\nVAD (Silero ONNX) load: {(time.perf_counter() - t0) * 1000:.0f} ms")
    chunks = [audio[i:i + 512] for i in range(0, len(audio) - 511, 512)]
    for c in chunks[:20]:
        vad.process_chunk(c)  # warm-up
    vad.reset()
    lat = [vad.process_chunk(chunks[i % len(chunks)]).latency_ms for i in range(n_chunks)]
    print("VAD per-chunk inference (512 samples = 32 ms):")
    _print("latency", percentiles(lat))
    print("  > 2 ms per chunk would block the event loop; AudioFrontend runs VAD on its own thread.")


def bench_asr(audio: np.ndarray, runs: int, model: str) -> None:
    from app.audio.asr import SpeechRecognizer
    from app.config import ASRConfig

    asr = SpeechRecognizer(ASRConfig(model=model))
    t0 = time.perf_counter()
    asr.load()
    print(f"\nASR ({model}) load: {(time.perf_counter() - t0) * 1000:.0f} ms")
    seconds = len(audio) / SR

    async def once() -> tuple[float, str]:
        # transcribe_async runs on the thread that loaded the model (MLX affinity).
        r = await asr.transcribe_async(audio, SR)
        return r.latency_ms, r.text

    try:
        cold_ms, text = asyncio.run(once())
        warm = [asyncio.run(once())[0] for _ in range(runs)]
    finally:
        asr.close()
    print(f"ASR on {seconds:.2f} s of audio:")
    print(f"  cold (first call)            latency={cold_ms:.1f} ms  RTF={cold_ms / 1000 / seconds:.3f}")
    _print("warm latency", percentiles(warm))
    _print("warm RTF", percentiles([ms / 1000 / seconds for ms in warm]), unit="")
    print(f'  transcript: "{text}"')


def bench_endpointing(trials: int = 50) -> None:
    """Speech-then-silence through the segmenter with scripted VAD decisions."""
    from app.audio.segmentation import SpeechSegmenter
    from app.audio.vad import VADResult

    cfg = load_config()
    chunk_s = cfg.audio.chunk_samples / cfg.audio.sample_rate
    speech = VADResult(is_speech=True, confidence=0.9, latency_ms=0.0)
    silence = VADResult(is_speech=False, confidence=0.1, latency_ms=0.0)
    chunk = np.zeros(cfg.audio.chunk_samples, dtype=np.float32)

    delays, per_chunk = [], []
    for trial in range(trials):
        seg = SpeechSegmenter(cfg.vad, cfg.audio.sample_rate, cfg.audio.chunk_samples)
        n_speech = 10 + trial  # 320 ms … ~1.9 s of speech
        pattern = [silence] * 5 + [speech] * n_speech + [silence] * 40
        for i, vr in enumerate(pattern):
            t0 = time.perf_counter()
            out = seg.process(chunk, vr, 1000.0 + (i + 1) * chunk_s)
            per_chunk.append((time.perf_counter() - t0) * 1000)
            if out is not None:
                delays.append((out.end_time - out.speech_end_time) * 1000)
                break
    print(f"\nEndpointing (speech_end_silence_ms={cfg.vad.speech_end_silence_ms}, "
          f"chunk={chunk_s * 1000:.0f} ms), capture-time delay from last speech chunk:")
    _print("endpoint delay", percentiles(delays))
    _print("segmenter cost per chunk", percentiles(per_chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", help="16-bit PCM WAV of speech to use for VAD and ASR")
    parser.add_argument("--asr-runs", type=int, default=10)
    parser.add_argument("--vad-chunks", type=int, default=2000)
    parser.add_argument("--skip-models", action="store_true", help="only the model-free endpointing benchmark")
    args = parser.parse_args()

    bench_endpointing()
    if args.skip_models:
        return
    cfg = load_config()
    audio = load_wav(args.wav) if args.wav else synthetic_speech(3.0)
    bench_vad(audio, args.vad_chunks, cfg.vad.threshold)
    bench_asr(audio, args.asr_runs, cfg.asr.model)


if __name__ == "__main__":
    main()
