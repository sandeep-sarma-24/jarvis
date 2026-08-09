"""Context Builder — assembles bounded context for the reasoning model.

Selects recent relevant working-memory entries, enforces character and message
limits, and ensures the current event is always preserved.

Phase 5 extends context with persistent conversation history and relevant memories.
Priority: current request > recent conversation > relevant memory.
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
    conversation_messages: tuple[dict[str, str], ...] = ()
    memory_context: tuple[str, ...] = ()
    visual_context: tuple[str, ...] = ()
    multimodal_context: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    total_characters: int = 0


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
    - Persistent conversation and memories are included within budget.
    """

    def __init__(
        self,
        *,
        max_messages: int = 8,
        max_characters: int = 6000,
        system_prompt_path: Path | None = None,
        max_memory_items: int = 5,
        max_visual_events: int = 5,
        max_visual_context_characters: int = 500,
    ) -> None:
        self._max_messages = max_messages
        self._max_characters = max_characters
        self._max_memory_items = max_memory_items
        self._max_visual_events = max_visual_events
        self._max_visual_chars = max_visual_context_characters
        self._system_prompt = _load_system_prompt(system_prompt_path)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def build(
        self,
        event: SpeechEvent,
        working_memory: WorkingMemory,
        *,
        conversation_messages: list[dict[str, str]] | None = None,
        memory_context: list[str] | None = None,
        visual_context: list[str] | None = None,
        multimodal_context: list[str] | None = None,
    ) -> Context:
        budget = self._max_characters
        budget -= len(self._system_prompt)
        budget -= len(event.transcript)

        # Phase 5: include persistent conversation messages within budget
        conv_msgs: list[dict[str, str]] = []
        if conversation_messages:
            for msg in conversation_messages:
                cost = len(msg.get("content", ""))
                if budget - cost < 0:
                    break
                conv_msgs.append(msg)
                budget -= cost

        # Phase 5: include relevant memories within budget
        mem_ctx: list[str] = []
        if memory_context:
            for mem_str in memory_context[: self._max_memory_items]:
                cost = len(mem_str)
                if budget - cost < 0:
                    break
                mem_ctx.append(mem_str)
                budget -= cost

        # Phase 8: multimodal context (fused audio + visual observations)
        # Takes priority over legacy visual_context when present.
        # Uses the same sub-budget as visual context.
        mm_ctx: list[str] = []
        vis_ctx: list[str] = []
        if multimodal_context:
            mm_budget = min(self._max_visual_chars, budget)
            mm_used = 0
            for obs in multimodal_context[: self._max_visual_events]:
                cost = len(obs)
                if mm_used + cost > mm_budget:
                    break
                mm_ctx.append(obs)
                mm_used += cost
            budget -= mm_used
        elif visual_context:
            # Phase 6 fallback: legacy visual-only context
            vis_budget = min(self._max_visual_chars, budget)
            vis_used = 0
            for desc in visual_context[: self._max_visual_events]:
                cost = len(desc)
                if vis_used + cost > vis_budget:
                    break
                vis_ctx.append(desc)
                vis_used += cost
            budget -= vis_used

        # Phase 1-4 working memory selection with remaining budget
        recent = self._select_recent(
            working_memory, budget, exclude_event_id=event.id,
        )
        total_chars = (
            len(self._system_prompt)
            + len(event.transcript)
            + sum(len(e.transcript) for e in recent)
            + sum(len(m.get("content", "")) for m in conv_msgs)
            + sum(len(s) for s in mem_ctx)
            + sum(len(s) for s in vis_ctx)
            + sum(len(s) for s in mm_ctx)
        )

        return Context(
            system_prompt=self._system_prompt,
            current_event=event,
            recent_events=tuple(recent),
            conversation_messages=tuple(conv_msgs),
            memory_context=tuple(mem_ctx),
            visual_context=tuple(vis_ctx),
            multimodal_context=tuple(mm_ctx),
            metadata={
                "context_built_at": time.time(),
                "recent_count": len(recent),
                "conversation_count": len(conv_msgs),
                "memory_count": len(mem_ctx),
                "visual_count": len(vis_ctx),
                "multimodal_count": len(mm_ctx),
                "total_characters": total_chars,
            },
            total_characters=total_chars,
        )

    def to_messages(self, ctx: Context) -> list[dict[str, str]]:
        """Convert Context into an OpenAI-style message list for the model.

        Order: system → memories → multimodal/visual context → conversation → working memory → current.
        """
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": ctx.system_prompt},
        ]

        # Relevant persistent memories as context
        if ctx.memory_context:
            memory_block = "Relevant information:\n" + "\n".join(
                f"- {m}" for m in ctx.memory_context
            )
            msgs.append({"role": "system", "content": memory_block})

        # Phase 8: fused multimodal context (audio + visual observations)
        if ctx.multimodal_context:
            mm_block = "Recent observations:\n" + "\n".join(
                f"- {obs}" for obs in ctx.multimodal_context
            )
            msgs.append({"role": "system", "content": mm_block})
        elif ctx.visual_context:
            # Phase 6 fallback: visual-only context
            visual_block = "Recent visual context:\n" + "\n".join(
                f"- {v}" for v in ctx.visual_context
            )
            msgs.append({"role": "system", "content": visual_block})

        # Persistent conversation history (already has roles)
        for msg in ctx.conversation_messages:
            msgs.append({"role": msg["role"], "content": msg["content"]})

        # Working memory entries (Phase 1-4 compatibility)
        for entry in ctx.recent_events:
            msgs.append({"role": "user", "content": entry.transcript})

        # Current request always last
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
