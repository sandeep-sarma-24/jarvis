"""ReasoningModel protocol and types.

The rest of JARVIS depends only on this protocol, never on a specific model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
