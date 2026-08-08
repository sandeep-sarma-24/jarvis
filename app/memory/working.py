"""Minimal bounded working memory for Phase 2.

In-process only — no persistence, no embeddings, no semantic retrieval.
Provides recent event history and last-interaction tracking for attention scoring.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from app.events.types import Event


@dataclass(frozen=True)
class WorkingMemoryEntry:
    event_id: str
    timestamp: float
    transcript: str
    should_reason: bool


class WorkingMemory:
    """Bounded ring of recent events for attention context.

    ponytail: this is a deque with a maxlen, not a sophisticated memory system.
    Upgrade path: Phase 5 adds multi-turn conversation tracking.
    """

    def __init__(self, max_size: int = 20) -> None:
        self._entries: deque[WorkingMemoryEntry] = deque(maxlen=max_size)
        self._max_size = max_size
        self._last_interaction_time: float | None = None
        self._last_should_reason: bool | None = None

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def entries(self) -> list[WorkingMemoryEntry]:
        return list(self._entries)

    @property
    def last_interaction_time(self) -> float | None:
        return self._last_interaction_time

    @property
    def last_should_reason(self) -> bool | None:
        return self._last_should_reason

    def record(self, event_id: str, transcript: str, should_reason: bool) -> None:
        now = time.time()
        self._entries.append(WorkingMemoryEntry(
            event_id=event_id,
            timestamp=now,
            transcript=transcript,
            should_reason=should_reason,
        ))
        if should_reason:
            self._last_interaction_time = now
        self._last_should_reason = should_reason

    def seconds_since_last_interaction(self) -> float | None:
        if self._last_interaction_time is None:
            return None
        return time.time() - self._last_interaction_time

    def clear(self) -> None:
        self._entries.clear()
        self._last_interaction_time = None
        self._last_should_reason = None

    def __len__(self) -> int:
        return len(self._entries)
