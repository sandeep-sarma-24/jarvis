"""Ollama-backed reasoning model.

Talks to the Ollama /api/chat endpoint. Synchronous: stream() is a generator
driven from a worker thread, generate() is called via asyncio.to_thread.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Iterator

import httpx

from app.reasoning.model import ReasoningError, ReasoningResult

logger = logging.getLogger(__name__)


class _ThinkFilter:
    """Drops inline <think>...</think> blocks from content that arrives in deltas.

    A tag may be split across deltas, so a trailing partial tag is held back
    until the next delta (or flush) decides what it is.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            tag = self.CLOSE if self._inside else self.OPEN
            i = self._buf.find(tag)
            if i >= 0:
                if not self._inside:
                    out.append(self._buf[:i])
                self._buf = self._buf[i + len(tag):]
                self._inside = not self._inside
                continue
            keep = next((k for k in range(len(tag) - 1, 0, -1) if self._buf.endswith(tag[:k])), 0)
            cut = len(self._buf) - keep
            if not self._inside:
                out.append(self._buf[:cut])
            self._buf = self._buf[cut:]
            return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest


def _socket_of(resp: httpx.Response):
    stream = resp.extensions.get("network_stream")
    return stream.get_extra_info("socket") if stream is not None else None


def _shutdown(sock) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _shutdown_on(cancel: threading.Event, done: threading.Event, sock) -> None:
    while not cancel.wait(0.05):
        if done.is_set():
            return
    if not done.is_set():
        _shutdown(sock)


def _stats(frame: dict, ttft_ms: float | None) -> dict[str, float]:
    """Ollama's final frame -> last_stats. Durations are reported in nanoseconds."""
    stats: dict[str, float] = {}
    if ttft_ms is not None:
        stats["ttft_ms"] = round(ttft_ms, 1)
    eval_count = frame.get("eval_count")
    if eval_count is not None:
        stats["output_tokens"] = eval_count
        if frame.get("eval_duration"):
            stats["tokens_per_second"] = round(eval_count / (frame["eval_duration"] / 1e9), 1)
    if frame.get("load_duration") is not None:
        stats["load_ms"] = round(frame["load_duration"] / 1e6, 1)
    if frame.get("prompt_eval_count") is not None:
        stats["prompt_tokens"] = frame["prompt_eval_count"]
    if frame.get("prompt_eval_duration") is not None:
        stats["prompt_eval_ms"] = round(frame["prompt_eval_duration"] / 1e6, 1)
    return stats


def format_residency(r: dict | None) -> str:
    """One line from residency(): name, size, VRAM share (Metal on Apple Silicon), context."""
    if not r:
        return "not loaded"
    size, vram = r.get("size") or 0, r.get("size_vram") or 0
    gpu = f"{100 * vram / size:.0f}%" if size else "?"
    return (f"{r.get('name')} size {size / 2**30:.2f} GiB, size_vram {vram / 2**30:.2f} GiB "
            f"({gpu} GPU), context {r.get('context_length', '?')}")


class OllamaModel:
    """StreamingReasoningModel backed by one Ollama host."""

    def __init__(
        self,
        *,
        model: str = "gemma4:12b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 2.0,
        keep_alive: str = "30m",
        think: bool = False,
        num_ctx: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._keep_alive = keep_alive
        self._think = think
        self._num_ctx = num_ctx
        # timeout_seconds bounds each read (gap between stream lines), not the whole answer.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds),
            transport=transport,
        )
        self._last_stats: dict[str, float] = {}
        self._lock = threading.Lock()
        self._socks: set = set()  # sockets of in-flight streams, so close() can abort them

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def last_stats(self) -> dict[str, float]:
        return self._last_stats

    def _options(self, **options) -> dict:
        # Every request, warm-up included, must send the same num_ctx or Ollama reloads the model.
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        return options

    def _payload(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, stream: bool) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self._keep_alive,
            "think": self._think,
            "options": self._options(num_predict=max_tokens, temperature=temperature),
        }

    def _unreachable(self, exc: Exception) -> ReasoningError:
        return ReasoningError(
            f"{self._model}: {type(exc).__name__}: cannot reach Ollama at {self._base_url} ({exc}). "
            f"Start the local server with `ollama serve` and run `ollama pull {self._model}`."
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        cancel: threading.Event | None = None,
    ) -> Iterator[str]:
        self._last_stats = {}
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        think = _ThinkFilter()
        done = threading.Event()
        sock = None
        try:
            # Leaving this block (return, exception, or generator close) closes the response,
            # which aborts the request on the server side.
            with self._client.stream(
                "POST", f"{self._base_url}/api/chat",
                json=self._payload(messages, max_tokens, temperature, True),
            ) as resp:
                sock = _socket_of(resp)
                if sock is not None:
                    with self._lock:
                        self._socks.add(sock)
                    if cancel is not None:
                        # A blocked read (model load, long prompt eval) only sees `cancel`
                        # when the next line arrives; shutting the socket unblocks it now.
                        threading.Thread(target=_shutdown_on, args=(cancel, done, sock), daemon=True).start()
                if resp.status_code >= 400:
                    resp.read()
                    hint = f" Run `ollama pull {self._model}`." if resp.status_code == 404 else ""
                    raise ReasoningError(f"{self._model}: HTTP {resp.status_code}: {resp.text[:200]}{hint}")
                for line in resp.iter_lines():
                    if cancel is not None and cancel.is_set():
                        return
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line)
                    except ValueError as exc:
                        raise ReasoningError(f"{self._model}: malformed stream line {line[:100]!r}") from exc
                    if frame.get("error"):
                        raise ReasoningError(f"{self._model}: {frame['error']}")
                    # message.thinking is deliberately ignored.
                    text = think.feed((frame.get("message") or {}).get("content") or "")
                    if frame.get("done"):
                        text += think.flush()
                    if ttft_ms is None:
                        text = text.lstrip()  # whitespace left over from a stripped think block
                    if text:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        yield text
                    if frame.get("done"):
                        self._last_stats = _stats(frame, ttft_ms)
                        return
            if cancel is not None and cancel.is_set():
                return
            raise ReasoningError(f"{self._model}: stream ended without done frame")
        except ReasoningError:
            raise
        except Exception as exc:  # httpx errors, or a socket shut by cancel / close()
            if cancel is not None and cancel.is_set():
                return
            if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                raise self._unreachable(exc) from exc
            raise ReasoningError(f"{self._model}: {type(exc).__name__}: {exc}") from exc
        finally:
            done.set()
            with self._lock:
                self._socks.discard(sock)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> ReasoningResult:
        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                f"{self._base_url}/api/chat",
                json=self._payload(messages, max_tokens, temperature, False),
            )
            resp.raise_for_status()
            data = resp.json()

            think = _ThinkFilter()
            text = think.feed(data.get("message", {}).get("content", "")) + think.flush()
            latency = (time.perf_counter() - t0) * 1000

            return ReasoningResult(
                text=text.strip(),
                model=self._model,
                latency_ms=round(latency, 1),
                success=True,
                input_tokens=data.get("prompt_eval_count"),
                output_tokens=data.get("eval_count"),
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return ReasoningResult(
                text="",
                model=self._model,
                latency_ms=round(latency, 1),
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def warm_up(self) -> bool:
        """Load the model (an empty chat loads it without generating). Never raises.

        Returns whether the host answered, so the fallback wrapper can skip a dead primary.
        """
        try:
            resp = self._client.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": [], "keep_alive": self._keep_alive, "stream": False,
                      "options": self._options()},
                # A cold load can outlast the per-read timeout; a dead host still fails at connect.
                timeout=httpx.Timeout(120.0, connect=self._client.timeout.connect),
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("reasoning warm-up failed for %s at %s: %s", self._model, self._base_url, exc)
            return False
        logger.info("reasoning model %s warm at %s: %s", self._model, self._base_url,
                    format_residency(self.residency()))
        return True

    def residency(self) -> dict | None:
        """This model's entry in GET /api/ps (name, size, size_vram, context_length), or None. Never raises."""
        try:
            resp = self._client.get(f"{self._base_url}/api/ps")
            resp.raise_for_status()
            for m in resp.json().get("models", []):
                if self._model in (m.get("name"), m.get("model")) or m.get("name") == f"{self._model}:latest":
                    return {k: m[k] for k in ("name", "size", "size_vram", "context_length") if k in m}
        except Exception as exc:
            logger.debug("residency check failed for %s: %s", self._model, exc)
        return None

    def close(self) -> None:
        # Closing the client alone leaves a reader on another thread blocked until its read timeout.
        with self._lock:
            socks = list(self._socks)
        for s in socks:
            _shutdown(s)
        self._client.close()
