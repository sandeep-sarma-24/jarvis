"""Benchmark: reasoning pipeline latency and resource usage.

Requires Ollama running with the configured model pulled.
Run: python -m scripts.benchmark_reasoning

Measures:
- Model cold start (first inference)
- Warm inference latency
- Context building latency
- Tokens/sec (if reported by Ollama)
- RSS memory before/after model load / during inference
"""
from __future__ import annotations

import os
import statistics
import time

from app.config import load_config
from app.context.builder import ContextBuilder
from app.events.types import Modality, SpeechEvent
from app.memory.working import WorkingMemory


def _get_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(
        session_id="bench", source=Modality.AUDIO,
        transcript=text, duration_seconds=1.0,
    )


def _benchmark_context_builder() -> None:
    print("\n── Context Builder Benchmark ──\n")
    builder = ContextBuilder(max_messages=8, max_characters=6000)
    memory = WorkingMemory(max_size=20)

    for i in range(20):
        memory.record(f"e{i}", f"Previous event number {i} about various topics", should_reason=i % 2 == 0)

    event = _speech("Jarvis, why is this API returning 500 errors?")
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        ctx = builder.build(event, memory)
        _ = builder.to_messages(ctx)
        latencies.append((time.perf_counter() - t0) * 1000)

    print(f"  Context build (1000 iterations):")
    print(f"    mean:  {statistics.mean(latencies):.4f} ms")
    print(f"    p50:   {sorted(latencies)[500]:.4f} ms")
    print(f"    p95:   {sorted(latencies)[950]:.4f} ms")
    print(f"    p99:   {sorted(latencies)[990]:.4f} ms")
    print(f"    chars: {ctx.total_characters}")
    print(f"    msgs:  {len(ctx.recent_events) + 2}")  # system + recent + current

    # Context growth test
    print("\n── Context Growth Test ──\n")
    for event_count in [1, 5, 10, 20, 50, 100]:
        mem = WorkingMemory(max_size=100)
        for i in range(event_count):
            mem.record(f"e{i}", f"Event {i} with some longer content about things " * 3, should_reason=True)
        ctx = builder.build(_speech("current query"), mem)
        msgs = builder.to_messages(ctx)
        print(f"  {event_count:3d} events → {ctx.total_characters:5d} chars, {len(msgs):2d} messages")


def _benchmark_model() -> None:
    print("\n── Reasoning Model Benchmark ──\n")

    cfg = load_config()
    rss_before = _get_rss_mb()
    print(f"  RSS before model init: {rss_before:.0f} MB")

    from app.reasoning.local_model import OllamaModel

    model = OllamaModel(
        model=cfg.reasoning.model,
        base_url=cfg.reasoning.ollama_url,
        timeout_seconds=cfg.reasoning.timeout_seconds,
    )
    rss_after_init = _get_rss_mb()
    print(f"  RSS after model init:  {rss_after_init:.0f} MB")
    print(f"  Model: {model.model_name}")

    messages = [
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    # Cold start
    print("\n  Cold start:")
    result = model.generate(messages, max_tokens=50, temperature=0.0)
    if not result.success:
        print(f"    FAILED: {result.error}")
        print("    Skipping warm inference benchmark.")
        model.close()
        return
    print(f"    latency:  {result.latency_ms:.0f} ms")
    print(f"    response: {result.text!r}")
    if result.output_tokens:
        tok_s = result.output_tokens / (result.latency_ms / 1000) if result.latency_ms > 0 else 0
        print(f"    tokens:   {result.output_tokens} ({tok_s:.1f} tok/s)")

    rss_during = _get_rss_mb()
    print(f"    RSS during inference: {rss_during:.0f} MB")

    # Warm inference (5 iterations)
    print("\n  Warm inference (5 iterations):")
    prompts = [
        "Explain what a function is in one sentence.",
        "What does HTTP 500 mean?",
        "Name the capital of France.",
        "What is Python used for?",
        "Explain what an API is briefly.",
    ]

    latencies = []
    tokens_list = []
    for prompt in prompts:
        msgs = [
            {"role": "system", "content": "Answer in one sentence."},
            {"role": "user", "content": prompt},
        ]
        r = model.generate(msgs, max_tokens=100, temperature=0.0)
        if r.success:
            latencies.append(r.latency_ms)
            if r.output_tokens:
                tokens_list.append(r.output_tokens)
            print(f"    {r.latency_ms:7.0f}ms | {r.text[:60]}...")
        else:
            print(f"    FAILED: {r.error}")

    if latencies:
        print(f"\n    mean latency:  {statistics.mean(latencies):.0f} ms")
        print(f"    min latency:   {min(latencies):.0f} ms")
        print(f"    max latency:   {max(latencies):.0f} ms")
    if tokens_list:
        total_tokens = sum(tokens_list)
        total_time_s = sum(latencies) / 1000
        print(f"    total tokens:  {total_tokens}")
        print(f"    overall tok/s: {total_tokens / total_time_s:.1f}")

    model.close()


def main() -> None:
    _benchmark_context_builder()
    try:
        _benchmark_model()
    except Exception as e:
        print(f"\n  Model benchmark skipped: {e}")
        print("  (Is Ollama running with the model pulled?)")


if __name__ == "__main__":
    main()
