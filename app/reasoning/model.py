"""ReasoningModel protocol and types.

The rest of JARVIS depends only on this protocol, never on a specific model.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class ReasoningResult:
    text: str
    model: str
    latency_ms: float
    success: bool
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class ReasoningModel(Protocol):
    """Abstract interface for reasoning models.

    Implementations must be synchronous — the caller is responsible for
    running inference off the event loop via asyncio.to_thread.
    """

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> ReasoningResult: ...


class ReasoningError(RuntimeError):
    """Raised by stream() when generation fails before or during the stream."""


@runtime_checkable
class StreamingReasoningModel(ReasoningModel, Protocol):
    """A ReasoningModel that can also stream. generate() stays the fallback.

    stream() is a synchronous generator, driven from a worker thread:
    - yields user-visible answer text deltas only (never thinking content);
    - raises ReasoningError on failure, before or mid-stream;
    - returns promptly once `cancel` is set, aborting the underlying request;
    - closing the generator also aborts the request.
    - `tools` (Ollama schemas) is honoured only by backends that support tool calling; their calls
      arrive via on_tool_calls (called from the worker thread) and never as text.
    After the stream ends, last_stats holds what the backend reported
    (keys when known: ttft_ms, output_tokens, tokens_per_second, load_ms).
    """

    @property
    def last_stats(self) -> dict[str, float]: ...

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        cancel: threading.Event | None = None,
        tools: list[dict] | None = None,
        on_tool_calls: Callable[[list[dict]], None] | None = None,
    ) -> Iterator[str]: ...

    def warm_up(self) -> None:
        """Load the model into memory ahead of the first request. Blocking; never raises."""
        ...

    def close(self) -> None: ...
