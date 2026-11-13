"""Benchmark: local reasoning latency per context window and prompt size, plus context-building cost.

Run:
  python -m scripts.benchmark_reasoning                      # configured primary (+ fallback unless local_only)
  python -m scripts.benchmark_reasoning --runs 20 --num-ctx 4096 --num-ctx 8192 --json out.json
  python -m scripts.benchmark_reasoning --target url=http://127.0.0.1:11434,model=gemma4:e2b

Per target and per --num-ctx (the model is unloaded with keep_alive=0 before each, so each
context size starts with a fresh load):
- cold request: one short prompt straight after the unload (pays the model load)
- model residency from /api/ps: size, size_vram (GPU/Metal share), context length
- per prompt size (short ~20 tokens, medium ~500, long ~2500 of conversation history):
  TTFT, total, tokens/sec, prompt_eval_ms, prompt tokens over --runs warm requests
- peak RSS of all "ollama" processes, sampled every 0.2 s across the run
- macOS memory pressure (free %) and swap used at the end of each context size
The history is rotated per run so Ollama's prompt-prefix cache does not hide prompt processing.
P50/P95/P99 via app.contracts.percentiles. Unreachable targets print NOT RUN.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time

import httpx
import psutil

from app.config import load_config
from app.context.builder import ContextBuilder
from app.contracts import percentiles
from app.events.types import Modality, SpeechEvent
from app.memory.working import WorkingMemory
from app.reasoning.local_model import OllamaModel, format_residency
from app.reasoning.model import ReasoningError

SYSTEM = "You are a concise voice assistant. Answer in one or two short sentences."
PROMPTS = [
    "What does HTTP 500 mean?",
    "Name the capital of France.",
    "Explain what an API is.",
    "What is Python used for?",
    "Why is the sky blue?",
]
# (user, assistant) turns; repeated to build the medium and long prompts. Keep in sync with
# scripts/benchmark_mlx_llm.py, which cannot import this module.
HISTORY = [
    ("Jarvis, what's on my calendar this afternoon?",
     "You have a design review at two and a call with the vendor at four."),
    ("Move the vendor call to tomorrow morning if it's free.",
     "Done. The vendor call is now tomorrow at nine thirty, and I sent them the update."),
    ("Why is the staging API returning 500 errors?",
     "The logs show the database connection pool is exhausted since the last deploy at noon."),
    ("Can you roll back that deploy?",
     "I rolled staging back to the previous build. Error rates are dropping to normal."),
    ("Remind me to write the incident summary before I leave today.",
     "I'll remind you at five thirty to write the incident summary."),
    ("What's the weather like for my run tomorrow?",
     "Tomorrow morning looks cool and dry, around twelve degrees with light wind."),
]
# Approximate prompt sizes in tokens (about 4 characters per token for English).
SIZES = {"short": 0, "medium": 500, "long": 2500}


def build_messages(size: str, question: str, rotate: int = 0) -> list[dict[str, str]]:
    """System prompt + about SIZES[size] tokens of history (rotated by `rotate`) + the question."""
    msgs = [{"role": "system", "content": SYSTEM}]
    chars, i = 0, rotate
    while chars < SIZES[size] * 4:
        user, assistant = HISTORY[i % len(HISTORY)]
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        chars += len(user) + len(assistant)
        i += 1
    return msgs + [{"role": "user", "content": question}]


def parse_memory_free_percent(text: str) -> float | None:
    m = re.search(r"System-wide memory free percentage:\s*([\d.]+)%", text)
    return float(m.group(1)) if m else None


def parse_swap_used_mb(text: str) -> float | None:
    m = re.search(r"used\s*=\s*([\d.]+)([KMG])", text)
    return float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2)] if m else None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def system_memory() -> dict[str, float | None]:
    return {"memory_free_percent": parse_memory_free_percent(_run(["memory_pressure"])),
            "swap_used_mb": parse_swap_used_mb(_run(["sysctl", "-n", "vm.swapusage"]))}


def ollama_rss_mb() -> float:
    """Sum of RSS over processes whose name contains "ollama" (server and model runners)."""
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        if "llama" in (p.info["name"] or "").lower() and p.info["memory_info"]:
            total += p.info["memory_info"].rss
    return total / 2**20


class PeakRss:
    """Samples ollama_rss_mb() on a thread while active; .peak_mb is the maximum seen."""

    def __init__(self, interval: float = 0.2) -> None:
        self.peak_mb = 0.0
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while True:
            self.peak_mb = max(self.peak_mb, ollama_rss_mb())
            if self._stop.wait(self._interval):
                return

    def __enter__(self) -> "PeakRss":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


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


def is_pulled(tags: dict, model_name: str) -> bool:
    """Whether GET /api/tags output lists `model_name` (an untagged name means :latest)."""
    pulled = {m.get("name") for m in tags.get("models", [])}
    return model_name in pulled or f"{model_name}:latest" in pulled


def _one(model: OllamaModel, messages: list[dict[str, str]], max_tokens: int) -> dict[str, float]:
    """One streamed request -> total_ms plus the backend's last_stats."""
    t0 = time.perf_counter()
    for _ in model.stream(messages, max_tokens=max_tokens, temperature=0.0):
        pass
    return {"total_ms": (time.perf_counter() - t0) * 1000, **model.last_stats}


def _fmt(label: str, values: list[float], unit: str) -> str:
    p = percentiles(values)
    if not p["n"]:
        return f"    {label:<14} n=0"
    return (f"    {label:<14} p50 {p['p50']:8.1f}  p95 {p['p95']:8.1f}  p99 {p['p99']:8.1f}  "
            f"min {p['min']:8.1f}  max {p['max']:8.1f} {unit}  (n={p['n']})")


METRICS = [("ttft_ms", "ttft", "ms"), ("total_ms", "total", "ms"), ("tokens_per_second", "tokens/sec", "tok/s"),
           ("prompt_eval_ms", "prompt eval", "ms"), ("prompt_tokens", "prompt tokens", "tok"),
           ("output_tokens", "output tokens", "tok")]


def summarize(rows: list[dict[str, float]]) -> dict[str, dict]:
    """Per-metric percentiles over the rows that report that metric."""
    return {key: percentiles([r[key] for r in rows if key in r]) for key, _, _ in METRICS}


def _unload(url: str, model_name: str) -> None:
    httpx.post(f"{url.rstrip('/')}/api/generate", json={"model": model_name, "keep_alive": 0},
               timeout=30).raise_for_status()


def _benchmark_ctx(url: str, model_name: str, num_ctx: int, runs: int, max_tokens: int) -> dict:
    print(f"\n  ── num_ctx {num_ctx} ──")
    cfg = load_config().reasoning
    model = OllamaModel(model=model_name, base_url=url, timeout_seconds=cfg.timeout_seconds,
                        connect_timeout_seconds=cfg.connect_timeout_seconds, keep_alive=cfg.keep_alive,
                        think=cfg.think, num_ctx=num_ctx)
    result: dict = {"num_ctx": num_ctx}
    try:
        # Unload so the first request pays the full model load at this context size.
        _unload(url, model_name)
        with PeakRss() as rss:
            try:
                cold = _one(model, build_messages("short", PROMPTS[0]), max_tokens)
                result["cold"] = cold
                print(f"  cold: total {cold['total_ms']:.0f} ms, ttft {cold.get('ttft_ms', float('nan')):.0f} ms, "
                      f"load {cold.get('load_ms', float('nan')):.0f} ms")
            except ReasoningError as exc:
                result["cold"] = {"error": str(exc)}
                print(f"  cold: FAILED {exc}")
            result["residency"] = model.residency()
            print(f"  residency: {format_residency(result['residency'])}")

            result["sizes"] = {}
            for size in SIZES:
                rows, failures = [], 0
                for i in range(runs):
                    try:
                        rows.append(_one(model, build_messages(size, PROMPTS[i % len(PROMPTS)], rotate=i), max_tokens))
                    except ReasoningError as exc:
                        failures += 1
                        print(f"    run {i}: FAILED {exc}")
                print(f"  {size} prompt, warm ({runs} runs, failure rate {failures / runs:.0%}):"
                      if runs else f"  {size}: 0 runs")
                for key, label, unit in METRICS:
                    print(_fmt(label, [r[key] for r in rows if key in r], unit))
                result["sizes"][size] = {"runs": runs, "failures": failures, "stats": summarize(rows), "rows": rows}
        result["ollama_peak_rss_mb"] = round(rss.peak_mb, 1)
        result.update(system_memory())
        print(f"  ollama peak RSS {result['ollama_peak_rss_mb']:.0f} MB, memory free "
              f"{result['memory_free_percent']}%, swap used {result['swap_used_mb']} MB")
    except httpx.HTTPError as exc:
        result["not_run"] = f"could not unload {model_name} ({exc})"
        print(f"  NOT RUN: {result['not_run']}")
    finally:
        model.close()
    return result


def _benchmark_target(url: str, model_name: str, num_ctxs: list[int], runs: int, max_tokens: int) -> dict:
    print(f"\n── {model_name} @ {url} ──")
    out: dict = {"url": url, "model": model_name}
    cfg = load_config().reasoning
    try:
        tags = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=cfg.connect_timeout_seconds + 3)
        tags.raise_for_status()
    except httpx.HTTPError as exc:
        out["not_run"] = f"{url} unreachable ({type(exc).__name__}: {exc})"
        print(f"  NOT RUN: {out['not_run']}")
        return out
    if not is_pulled(tags.json(), model_name):
        out["not_run"] = f"{model_name} is not pulled on {url} (run `ollama pull {model_name}`)"
        print(f"  NOT RUN: {out['not_run']}")
        return out
    out["contexts"] = [_benchmark_ctx(url, model_name, n, runs, max_tokens) for n in num_ctxs]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", action="append", type=_parse_target, default=[],
                        help="url=...,model=... (repeatable). Default: configured primary, plus the "
                             "fallback unless runtime.local_only.")
    parser.add_argument("--num-ctx", action="append", type=int, default=[],
                        help="context window to benchmark (repeatable). Default: reasoning.num_ctx.")
    parser.add_argument("--runs", type=int, default=20, help="warm requests per prompt size")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--json", metavar="PATH", help="also write all results as JSON")
    args = parser.parse_args()

    full = load_config()
    cfg = full.reasoning
    targets = args.target
    if not targets:
        targets = [(cfg.ollama_url, cfg.model)]
        if cfg.fallback_model and not full.runtime.local_only:
            targets.append((cfg.fallback_url, cfg.fallback_model))

    _benchmark_context_builder()
    print("\n── Reasoning Targets ──")
    results = [_benchmark_target(url, model_name, args.num_ctx or [cfg.num_ctx], args.runs, args.max_tokens)
               for url, model_name in targets]
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"max_tokens": args.max_tokens, "targets": results}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
