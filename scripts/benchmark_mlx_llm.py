"""Benchmark: native MLX (mlx-lm) text generation, for comparison with scripts/benchmark_reasoning.py.

Run in a venv that has only mlx-lm and psutil (this file imports nothing from `app`):
  python scripts/benchmark_mlx_llm.py --runs 20 --json mlx.json
  python scripts/benchmark_mlx_llm.py --model mlx-community/gemma-4-E2B-it-qat-4bit --max-tokens 256

Measures: cold load time, a first-generation warm-up, then per prompt size (short ~20 tokens,
medium ~500, long ~2500 of conversation history; same system prompt, questions and history as
benchmark_reasoning) TTFT P50/P95/P99, generation tokens/s, prompt tokens/s, prompt tokens,
MLX peak memory, process RSS, plus macOS memory pressure and swap used at the end.
TTFT = time from calling stream_generate to the first non-empty text segment. No prompt cache is
used, so every run processes its full prompt.

Gemma 4 checkpoints: written against mlx-lm 0.32.2, whose mlx_lm/models/gemma4.py loads the
multimodal "gemma4" checkpoint as text-only (sanitize() drops the vision/audio towers, and
"gemma4_unified" is remapped to it), so mlx-vlm should not be needed for text. If an installed
mlx-lm lacks gemma4 support, load fails with "Model type gemma4 not supported" and this script
says so; upgrade mlx-lm (or use mlx-vlm) in that case.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time

import psutil

# ── Copied from scripts/benchmark_reasoning.py (cannot import app); keep in sync. ──
SYSTEM = "You are a concise voice assistant. Answer in one or two short sentences."
PROMPTS = [
    "What does HTTP 500 mean?",
    "Name the capital of France.",
    "Explain what an API is.",
    "What is Python used for?",
    "Why is the sky blue?",
]
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
SIZES = {"short": 0, "medium": 500, "long": 2500}


def build_messages(size: str, question: str, rotate: int = 0) -> list[dict[str, str]]:
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


def percentiles(values: list[float]) -> dict[str, float]:
    """Nearest-rank, same as app.contracts.percentiles."""
    if not values:
        return {"n": 0}
    s = sorted(values)

    def rank(p: float) -> float:
        return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]

    return {"n": len(s), "min": s[0], "p50": rank(0.50), "p95": rank(0.95), "p99": rank(0.99), "max": s[-1]}
# ── end of copied block ──


def _fmt(label: str, values: list[float], unit: str) -> str:
    p = percentiles(values)
    if not p["n"]:
        return f"    {label:<14} n=0"
    return (f"    {label:<14} p50 {p['p50']:8.1f}  p95 {p['p95']:8.1f}  p99 {p['p99']:8.1f}  "
            f"min {p['min']:8.1f}  max {p['max']:8.1f} {unit}  (n={p['n']})")


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def _one(model, tokenizer, messages, max_tokens: int, stream_generate, sampler) -> dict[str, float]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The chat template already has the BOS token (mirrors mlx_lm.generate's CLI).
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    t0 = time.perf_counter()
    ttft_ms, last, text = None, None, []
    for resp in stream_generate(model, tokenizer, tokens, max_tokens=max_tokens, sampler=sampler):
        if ttft_ms is None and resp.text:
            ttft_ms = (time.perf_counter() - t0) * 1000
        text.append(resp.text)
        last = resp
    row = {"total_ms": (time.perf_counter() - t0) * 1000, "rss_mb": _rss_mb()}
    if ttft_ms is not None:
        row["ttft_ms"] = ttft_ms
    if last is not None:
        row.update(prompt_tokens=last.prompt_tokens, prompt_tps=last.prompt_tps,
                   output_tokens=last.generation_tokens, tokens_per_second=last.generation_tps,
                   peak_memory_gb=last.peak_memory)
    row["text"] = "".join(text)
    return row


METRICS = [("ttft_ms", "ttft", "ms"), ("total_ms", "total", "ms"), ("tokens_per_second", "gen tok/s", "tok/s"),
           ("prompt_tps", "prompt tok/s", "tok/s"), ("prompt_tokens", "prompt tokens", "tok"),
           ("output_tokens", "output tokens", "tok"), ("peak_memory_gb", "mlx peak mem", "GB"),
           ("rss_mb", "process RSS", "MB")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mlx-community/gemma-4-E2B-it-qat-4bit")
    parser.add_argument("--runs", type=int, default=20, help="warm generations per prompt size")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", metavar="PATH", help="also write all results as JSON")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm import __version__ as mlx_lm_version
    except ImportError as exc:
        print(f"NOT RUN: mlx-lm is not installed in this environment ({exc}). pip install mlx-lm psutil")
        return 1

    out: dict = {"model": args.model, "mlx_lm_version": mlx_lm_version, "max_tokens": args.max_tokens}
    print(f"── MLX {args.model} (mlx-lm {mlx_lm_version}) ──\n")
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    try:
        model, tokenizer = load(args.model)
    except Exception as exc:  # unsupported model type, missing weights, download failure
        print(f"FAILED to load {args.model}: {type(exc).__name__}: {exc}")
        if "not supported" in str(exc):
            print("  this mlx-lm cannot load this architecture; upgrade mlx-lm, or try mlx-vlm for this checkpoint")
        return 1
    out["load_s"] = round(time.perf_counter() - t0, 2)
    out["rss_after_load_mb"] = round(_rss_mb(), 1)
    print(f"  load (cold): {out['load_s']:.2f} s, RSS {rss0:.0f} -> {out['rss_after_load_mb']:.0f} MB")
    sampler = make_sampler(temp=args.temperature)

    try:
        warm = _one(model, tokenizer, build_messages("short", PROMPTS[0]), args.max_tokens, stream_generate, sampler)
    except Exception as exc:
        print(f"FAILED first generation: {type(exc).__name__}: {exc}")
        return 1
    out["first_generation"] = warm
    print(f"  first generation: total {warm['total_ms']:.0f} ms, ttft {warm.get('ttft_ms', float('nan')):.0f} ms, "
          f"reply {warm['text'][:60]!r}")

    out["sizes"] = {}
    for size in SIZES:
        mx.reset_peak_memory()
        rows = [_one(model, tokenizer, build_messages(size, PROMPTS[i % len(PROMPTS)], rotate=i),
                     args.max_tokens, stream_generate, sampler) for i in range(args.runs)]
        print(f"\n  {size} prompt ({args.runs} runs):")
        for key, label, unit in METRICS:
            print(_fmt(label, [r[key] for r in rows if key in r], unit))
        out["sizes"][size] = {"stats": {k: percentiles([r[k] for r in rows if k in r]) for k, _, _ in METRICS},
                              "rows": rows}
    out["peak_rss_mb"] = round(max([_rss_mb()] + [r["rss_mb"] for s in out["sizes"].values() for r in s["rows"]]), 1)
    out.update(system_memory())
    print(f"\n  process RSS peak (sampled per run) {out['peak_rss_mb']:.0f} MB, memory free "
          f"{out['memory_free_percent']}%, swap used {out['swap_used_mb']} MB")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
