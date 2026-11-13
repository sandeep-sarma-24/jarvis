"""Verify the reasoning model answers from the local Ollama server on this Mac.

Run:
  python -m scripts.verify_local_gemma

Checks, in order: the configured reasoning URL(s) are loopback, the local server answers
GET /api/tags and lists the configured model, and one streamed prompt through
build_reasoning_model arrives in more than one chunk and contains "LOCAL OK".
Prints model, URL, TTFT, tokens/s, total latency and /api/ps residency. Exit 0 on success, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx

from app.config import ConfigError, is_loopback, load_config
from app.reasoning import build_reasoning_model
from app.reasoning.local_model import format_residency
from app.reasoning.model import ReasoningError
from scripts.benchmark_reasoning import is_pulled

PROMPT = "Reply with exactly: LOCAL OK"


def check_reply(text: str, chunks: int) -> str | None:
    """The failure message for a streamed reply, or None if it passes."""
    if chunks <= 1:
        return f"expected a streamed reply in more than one chunk, got {chunks}"
    if "local ok" not in text.lower():
        return f"reply does not contain LOCAL OK: {text!r}"
    return None


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def run() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        return _fail(f"config: {exc}")
    r = cfg.reasoning
    urls = [r.ollama_url] + ([r.fallback_url] if r.fallback_model else [])
    remote = [u for u in urls if not is_loopback(u)]
    if remote:
        return _fail(f"reasoning endpoint(s) not on this machine: {remote}. Use http://127.0.0.1:11434.")

    print(f"model {r.model} @ {r.ollama_url} (num_ctx {r.num_ctx})")
    try:
        tags = httpx.get(f"{r.ollama_url.rstrip('/')}/api/tags", timeout=r.connect_timeout_seconds + 3)
        tags.raise_for_status()
    except httpx.HTTPError as exc:
        return _fail(f"local Ollama not reachable at {r.ollama_url} ({type(exc).__name__}: {exc}). "
                     "Start it with `ollama serve`.")
    if not is_pulled(tags.json(), r.model):
        return _fail(f"{r.model} is not pulled. Run: ollama pull {r.model}")

    model = build_reasoning_model(r)
    try:
        model.warm_up()
        chunks, parts = 0, []
        t0 = time.perf_counter()
        try:
            for delta in model.stream([{"role": "user", "content": PROMPT}], max_tokens=16, temperature=0.0):
                chunks += 1
                parts.append(delta)
        except ReasoningError as exc:
            return _fail(str(exc))
        total_ms = (time.perf_counter() - t0) * 1000
        if model.model_name != r.model:  # FallbackModel answered with its secondary
            return _fail(f"answered by fallback {model.model_name}, not {r.model}")
        text = "".join(parts)
        stats = model.last_stats
        print(f"reply   {text.strip()!r} ({chunks} chunks)")
        print(f"ttft    {stats.get('ttft_ms', float('nan')):.0f} ms")
        print(f"tok/s   {stats.get('tokens_per_second', float('nan')):.1f}")
        print(f"total   {total_ms:.0f} ms")
        # residency() lives on OllamaModel; FallbackModel wraps it as _primary.
        print(f"resident {format_residency(getattr(model, '_primary', model).residency())}")
        err = check_reply(text, chunks)
        if err:
            return _fail(err)
    finally:
        model.close()
    print("PASS: reasoning runs on local Ollama")
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.exit(run())
