"""Primary/secondary reasoning backend with a cooldown on the primary."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterator

from app.reasoning.model import ReasoningError, ReasoningResult, StreamingReasoningModel

logger = logging.getLogger(__name__)


class FallbackModel:
    """Uses `primary` until it fails before its first token, then `secondary`.

    After such a failure the primary is skipped for `cooldown_seconds`. A failure
    after the first delta propagates: an answer is never restarted mid-way.
    """

    def __init__(
        self,
        primary: StreamingReasoningModel,
        secondary: StreamingReasoningModel,
        *,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._cooldown = cooldown_seconds
        self._skip_until = 0.0
        self._on_fallback = False
        self._active: StreamingReasoningModel = primary

    @property
    def model_name(self) -> str:
        return self._active.model_name

    @property
    def last_stats(self) -> dict[str, float]:
        return self._active.last_stats

    def _primary_due(self) -> bool:
        return time.monotonic() >= self._skip_until

    def _primary_failed(self, error: object) -> None:
        self._skip_until = time.monotonic() + self._cooldown
        if not self._on_fallback:
            self._on_fallback = True
            logger.warning(
                "reasoning: primary %s failed (%s); switching to fallback %s, retrying primary in %.0fs",
                self._primary.model_name, error, self._secondary.model_name, self._cooldown,
            )

    def _primary_ok(self) -> None:
        self._active = self._primary
        if self._on_fallback:
            self._on_fallback = False
            logger.info("reasoning: primary %s is back", self._primary.model_name)

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        cancel: threading.Event | None = None,
        tools: list[dict] | None = None,
        on_tool_calls: Callable[[list[dict]], None] | None = None,
    ) -> Iterator[str]:
        kwargs = {"max_tokens": max_tokens, "temperature": temperature, "cancel": cancel}
        if tools:  # each backend drops tools itself if it lacks the capability
            kwargs |= {"tools": tools, "on_tool_calls": on_tool_calls}
        if self._primary_due():
            gen = self._primary.stream(messages, **kwargs)
            try:
                first = next(gen)
            except StopIteration:  # empty answer or cancelled before the first token
                self._primary_ok()
                return
            except ReasoningError as exc:
                self._primary_failed(exc)
            else:
                self._primary_ok()
                try:
                    yield first
                    yield from gen
                finally:
                    gen.close()
                return
        self._active = self._secondary
        yield from self._secondary.stream(messages, **kwargs)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> ReasoningResult:
        if self._primary_due():
            result = self._primary.generate(messages, max_tokens=max_tokens, temperature=temperature)
            # Nothing has reached the user yet, so any failure is safe to retry on the secondary.
            if result.success:
                self._primary_ok()
                return result
            self._primary_failed(result.error)
        self._active = self._secondary
        return self._secondary.generate(messages, max_tokens=max_tokens, temperature=temperature)

    def warm_up(self) -> None:
        # Only load the secondary when the primary is down: on a 16 GB Mac it costs real memory.
        if self._primary.warm_up() is False:
            self._primary_failed("warm-up failed")
            self._secondary.warm_up()

    def close(self) -> None:
        self._primary.close()
        self._secondary.close()
