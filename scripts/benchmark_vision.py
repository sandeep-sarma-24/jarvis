#!/usr/bin/env python3
"""Phase 7 benchmark — measures local vision preprocessing and inference latencies.

Usage: python -m scripts.benchmark_vision

Uses [perception.vision] model, ollama_url, timeout_seconds and max_image_size from
configs/default.toml. Prints a NOT RUN line and exits 2 if the host is unreachable.

Reports preprocessing, inference, and end-to-end latencies as P50/P95/P99
(app.contracts.percentiles). Also verifies the model actually supports image input.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import time

import httpx
import numpy as np
from PIL import Image

from app.config import load_config
from app.contracts import percentiles

ITERATIONS = 10
VISION = load_config().perception.vision
MODEL = VISION.model
OLLAMA_URL = VISION.ollama_url


def _fmt_stats(label: str, values: list[float], unit: str = "ms") -> str:
    p = percentiles(values)
    if not p["n"]:
        return f"  {label}: no samples"
    return (
        f"  {label} (n={p['n']}):\n"
        + "\n".join(f"    {k}:  {p[k]:>8.1f} {unit}" for k in ("min", "p50", "p95", "p99", "max"))
    )


def _check_ollama_available() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _check_model_installed() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        models = resp.json().get("models", [])
        return any(m.get("name", "").startswith(MODEL.split(":")[0]) for m in models)
    except Exception:
        return False


def _check_vision_capable() -> bool:
    """Send a tiny image to the model and check if it responds without error."""
    img = Image.new("RGB", (4, 4), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "What is this?", "images": [b64]},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return len(content) > 0
    except Exception as e:
        print(f"  Vision capability check failed: {e}")
        return False


def _create_test_image(w: int = 640, h: int = 480) -> bytes:
    """Create a deterministic RGB test image."""
    arr = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
    return arr.tobytes()


def run() -> None:
    try:
        import psutil  # type: ignore[import-untyped]
        proc = psutil.Process(os.getpid())
        rss_start = proc.memory_info().rss / (1024 * 1024)
        have_psutil = True
    except ImportError:
        rss_start = 0
        have_psutil = False
        print("(psutil not installed — memory/CPU stats unavailable)\n")

    print(f"JARVIS Phase 7 Vision Benchmark")
    print(f"Model: {MODEL}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Iterations: {ITERATIONS}")
    if have_psutil:
        print(f"RSS at start: {rss_start:.1f} MB")
    print()

    # 1. Check Ollama
    print("1. Ollama availability...")
    if not _check_ollama_available():
        print(f"NOT RUN: Ollama not reachable at {OLLAMA_URL}")
        sys.exit(2)
    print("   PASS")

    # 2. Check model
    print(f"2. Model {MODEL} installed...")
    if not _check_model_installed():
        print(f"   FAIL — model {MODEL} not found")
        sys.exit(1)
    print("   PASS")

    # 3. Check vision capability
    print(f"3. Model {MODEL} vision-capable...")
    if not _check_vision_capable():
        print(f"   FAIL — model does not support image input")
        sys.exit(1)
    print("   PASS")

    # 4. Benchmark preprocessing
    print(f"\n4. Preprocessing benchmark ({ITERATIONS} iterations)...")
    from app.perception.local_vision import LocalVisionAnalyzer
    from app.perception.types import Frame

    class NullBackend:
        def infer(self, image_b64: str, prompt: str) -> str:
            return '{"event_type":"VISUAL_SUMMARY","description":"benchmark","confidence":0.5}'

    analyzer = LocalVisionAnalyzer(backend=NullBackend(), max_image_size=VISION.max_image_size)
    frame_data = _create_test_image(640, 480)
    frame = Frame(
        timestamp=time.time(), data=frame_data, width=640, height=480, format="rgb24",
    )

    preprocess_times: list[float] = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        analyzer._preprocess(frame)
        preprocess_times.append((time.perf_counter() - t0) * 1000)

    print(_fmt_stats("Preprocessing", preprocess_times))

    # 5. Benchmark full inference
    print(f"\n5. Full inference benchmark ({ITERATIONS} iterations)...")
    from app.perception.local_vision import OllamaVisionBackend

    backend = OllamaVisionBackend(
        model=MODEL, base_url=OLLAMA_URL, timeout_seconds=VISION.timeout_seconds,
    )
    real_analyzer = LocalVisionAnalyzer(
        backend=backend, timeout_seconds=VISION.timeout_seconds, max_image_size=VISION.max_image_size,
    )

    inference_times: list[float] = []
    e2e_times: list[float] = []
    successes = 0

    for i in range(ITERATIONS):
        frame = Frame(
            timestamp=time.time(), data=frame_data, width=640, height=480, format="rgb24",
        )
        t0 = time.perf_counter()
        result = asyncio.run(real_analyzer.analyze(frame))
        elapsed = (time.perf_counter() - t0) * 1000
        e2e_times.append(elapsed)
        inference_times.append(real_analyzer.inference_latency_ms)

        if result is not None:
            successes += 1
            if i == 0:
                print(f"   Sample output: event_type={result.event_type.name} "
                      f"desc={result.description[:60]!r} conf={result.confidence:.2f}")

        print(f"   [{i+1}/{ITERATIONS}] {elapsed:.0f}ms {'OK' if result else 'FAIL'}")

    print(f"\n   Success rate: {successes}/{ITERATIONS}")
    print(_fmt_stats("Inference (model only)", inference_times))
    print(_fmt_stats("End-to-end", e2e_times))

    backend.close()
    real_analyzer.close()

    # 6. Memory/CPU
    if have_psutil:
        rss_end = proc.memory_info().rss / (1024 * 1024)
        cpu = proc.cpu_percent(interval=0.5)
        print(f"\n6. Resources:")
        print(f"   RSS at end: {rss_end:.1f} MB (delta: {rss_end - rss_start:+.1f} MB)")
        print(f"   CPU: {cpu:.1f}%")
    else:
        print(f"\n6. Resources: (psutil unavailable)")

    # GPU: best-effort nvidia-smi
    print(f"\n   GPU: check manually with `nvidia-smi` or `ollama ps`")

    print(f"\nDone.")


if __name__ == "__main__":
    run()
