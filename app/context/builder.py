"""Context Builder — assembles bounded context for the reasoning model.

Selects recent relevant working-memory entries, enforces character and message
limits, and ensures the current event is always preserved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.events.types import SpeechEvent
from app.memory.working import WorkingMemory, WorkingMemoryEntry

_DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "system_prompt.txt"
)


@dataclass(frozen=True)
class Context:
    system_prompt: str
    current_event: SpeechEvent
    recent_events: tuple[WorkingMemoryEntry, ...]
    metadata: dict[str, object]
    total_characters: int


def _load_system_prompt(path: Path | None = None) -> str:
    p = path or _DEFAULT_SYSTEM_PROMPT_PATH
    if p.exists():
        return p.read_text().strip()
    return "You are JARVIS, a local personal AI assistant. Be concise."


class ContextBuilder:
    """Assembles bounded context for the reasoning model.

    Guarantees:
    - Current event transcript is never truncated.
    - Total context stays within max_characters.
    - At most max_messages recent entries are included.
    - Recent meaningful interactions are preferred.
    """

    def __init__(
        self,
        *,
        max_messages: int = 8,
        max_characters: int = 6000,
        system_prompt_path: Path | None = None,
    ) -> None:
        self._max_messages = max_messages
        self._max_characters = max_characters
        self._system_prompt = _load_system_prompt(system_prompt_path)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def build(
        self,
        event: SpeechEvent,
        working_memory: WorkingMemory,
    ) -> Context:
        budget = self._max_characters
        budget -= len(self._system_prompt)
        budget -= len(event.transcript)

        recent = self._select_recent(
            working_memory, budget, exclude_event_id=event.id,
        )
        total_chars = (
            len(self._system_prompt)
            + len(event.transcript)
            + sum(len(e.transcript) for e in recent)
        )

        return Context(
            system_prompt=self._system_prompt,
            current_event=event,
            recent_events=tuple(recent),
            metadata={
                "context_built_at": time.time(),
                "recent_count": len(recent),
                "total_characters": total_chars,
            },
            total_characters=total_chars,
        )

    def to_messages(self, ctx: Context) -> list[dict[str, str]]:
        """Convert Context into an OpenAI-style message list for the model.

        All working-memory entries are user speech in Phase 3.
        ponytail: Phase 5 adds JARVIS response tracking in working memory,
        which will require explicit conversation role distinction here.
        """
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": ctx.system_prompt},
        ]
        for entry in ctx.recent_events:
            msgs.append({"role": "user", "content": entry.transcript})
        msgs.append({"role": "user", "content": ctx.current_event.transcript})
        return msgs

    def _select_recent(
        self,
        memory: WorkingMemory,
        char_budget: int,
        *,
        exclude_event_id: str | None = None,
    ) -> list[WorkingMemoryEntry]:
        entries = [
            e for e in memory.entries
            if e.event_id != exclude_event_id
        ]
        # Prefer meaningful interactions, most recent first
        candidates = [e for e in reversed(entries) if e.should_reason]
        if len(candidates) < self._max_messages:
            non_reason = [e for e in reversed(entries) if not e.should_reason]
            candidates.extend(non_reason)

        selected: list[WorkingMemoryEntry] = []
        used = 0
        for entry in candidates[: self._max_messages]:
            cost = len(entry.transcript)
            if used + cost > char_budget:
                break
            selected.append(entry)
            used += cost

        selected.reverse()
        return selected
