"""Benchmark: reasoning latency across Ollama targets, plus context-building cost.

Run:
  python -m scripts.benchmark_reasoning                      # targets from config (primary + fallback)
  python -m scripts.benchmark_reasoning --runs 20 \\
      --target url=http://127.0.0.1:11434,model=gemma4:12b \\
      --target url=http://gpu-host:11434,model=gemma4:e2b-it-qat

Per target, over the streaming path the assistant uses:
- cold request: the model is unloaded first (keep_alive=0), then one request
- warm TTFT, total latency, tokens/sec from Ollama's eval_duration
- failure rate
P50/P95/P99 via app.contracts.percentiles. Unreachable targets print NOT RUN.
"""
from __future__ import annotations

import argparse
import statistics
import time

import httpx

from app.config import load_config
from app.context.builder import ContextBuilder
from app.contracts import percentiles
from app.events.types import Modality, SpeechEvent
from app.memory.working import WorkingMemory
from app.reasoning.local_model import OllamaModel
from app.reasoning.model import ReasoningError

SYSTEM = "You are a concise voice assistant. Answer in one or two short sentences."
PROMPTS = [
    "What does HTTP 500 mean?",
    "Name the capital of France.",
    "Explain what an API is.",
    "What is Python used for?",
    "Why is the sky blue?",
]


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


def _parse_target(spec: str) -> tuple[str, str]:
    fields = dict(part.split("=", 1) for part in spec.split(","))
    if set(fields) != {"url", "model"}:
        raise argparse.ArgumentTypeError(f"expected url=...,model=..., got {spec!r}")
    return fields["url"], fields["model"]


def _one(model: OllamaModel, prompt: str, max_tokens: int) -> dict[str, float]:
    """One streamed request -> total_ms plus the backend's last_stats."""
    t0 = time.perf_counter()
    for _ in model.stream([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                          max_tokens=max_tokens, temperature=0.0):
        pass
    return {"total_ms": (time.perf_counter() - t0) * 1000, **model.last_stats}


def _fmt(label: str, values: list[float], unit: str) -> str:
    p = percentiles(values)
    if not p["n"]:
        return f"    {label:<14} n=0"
    return (f"    {label:<14} p50 {p['p50']:8.1f}  p95 {p['p95']:8.1f}  p99 {p['p99']:8.1f}  "
            f"min {p['min']:8.1f}  max {p['max']:8.1f} {unit}  (n={p['n']})")


def _benchmark_target(url: str, model_name: str, runs: int, max_tokens: int) -> None:
    print(f"\n── {model_name} @ {url} ──\n")
    cfg = load_config().reasoning
    try:
        tags = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=cfg.connect_timeout_seconds + 3)
        tags.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  NOT RUN: {url} unreachable ({type(exc).__name__}: {exc})")
        return
    pulled = {m.get("name") for m in tags.json().get("models", [])}
    if model_name not in pulled and f"{model_name}:latest" not in pulled:
        print(f"  NOT RUN: {model_name} is not pulled on {url}")
        return

    model = OllamaModel(model=model_name, base_url=url, timeout_seconds=cfg.timeout_seconds,
                        connect_timeout_seconds=cfg.connect_timeout_seconds, keep_alive=cfg.keep_alive,
                        think=cfg.think)
    try:
        # Unload so the first request pays the full model load.
        httpx.post(f"{url.rstrip('/')}/api/generate", json={"model": model_name, "keep_alive": 0},
                   timeout=30).raise_for_status()
        try:
            cold = _one(model, PROMPTS[0], max_tokens)
            print(f"  cold: total {cold['total_ms']:.0f} ms, ttft {cold.get('ttft_ms', float('nan')):.0f} ms, "
                  f"load {cold.get('load_ms', float('nan')):.0f} ms")
        except ReasoningError as exc:
            print(f"  cold: FAILED {exc}")

        rows, failures = [], 0
        for i in range(runs):
            try:
                rows.append(_one(model, PROMPTS[i % len(PROMPTS)], max_tokens))
            except ReasoningError as exc:
                failures += 1
                print(f"    run {i}: FAILED {exc}")
        print(f"  warm ({runs} runs, failure rate {failures / runs:.0%}):" if runs else "  warm: 0 runs")
        print(_fmt("ttft", [r["ttft_ms"] for r in rows if "ttft_ms" in r], "ms"))
        print(_fmt("total", [r["total_ms"] for r in rows], "ms"))
        print(_fmt("tokens/sec", [r["tokens_per_second"] for r in rows if "tokens_per_second" in r], "tok/s"))
        print(_fmt("output tokens", [r["output_tokens"] for r in rows if "output_tokens" in r], "tok"))
    except httpx.HTTPError as exc:
        print(f"  NOT RUN: could not unload {model_name} ({exc})")
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", action="append", type=_parse_target, default=[],
                        help="url=...,model=... (repeatable). Default: configured primary and fallback.")
    parser.add_argument("--runs", type=int, default=20, help="warm requests per target")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    targets = args.target
    if not targets:
        cfg = load_config().reasoning
        targets = [(cfg.ollama_url, cfg.model)]
        if cfg.fallback_model:
            targets.append((cfg.fallback_url, cfg.fallback_model))

    _benchmark_context_builder()
    print("\n── Reasoning Targets ──")
    for url, model_name in targets:
        _benchmark_target(url, model_name, args.runs, args.max_tokens)


if __name__ == "__main__":
    main()
