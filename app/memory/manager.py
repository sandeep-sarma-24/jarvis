"""Memory Manager — coordinates conversation, persistent memory, and retrieval."""
from __future__ import annotations

import logging
import re
import time

from app.memory.conversation import ConversationState
from app.memory.sqlite_store import SQLiteStore, normalize
from app.memory.store import ConversationStore, MemoryStore
from app.memory.types import ConversationSession, Memory, MemoryType

logger = logging.getLogger(__name__)

# Explicit "remember" requests. Deterministic and conservative: anything that
# does not match one of these shapes is ignored.
_LEAD = re.compile(
    r"^(?:(?:hey|ok|okay)\s+)?(?:jarvis\b[\s,.:;!-]*)?"
    r"(?:(?:please|can you|could you|would you|will you)\s+)?",
    re.IGNORECASE,
)
_FORGET = r"(?:don't|dont|do not)\s+forget"
_REMEMBER_RULES = (
    (re.compile(rf"^(?:remember|note|make a note|take a note|{_FORGET})\s+that\s+(?P<fact>.+)$", re.I),
     MemoryType.FACT, "{}"),
    (re.compile(rf"^(?:remember|{_FORGET})\s+(?P<fact>(?:my|our|the)\s+\S.*?\s(?:is|are|was|were)\s+\S.*)$", re.I),
     MemoryType.FACT, "{}"),
    (re.compile(rf"^{_FORGET}\s+to\s+(?P<fact>.+)$", re.I), MemoryType.TASK, "Reminder: {}"),
)
# "remember that time we ..." is reminiscing, not a request to store.
_NOT_A_FACT = re.compile(r"^(?:time|day|night|one|when|place)\b", re.I)
_EXPLICIT_IMPORTANCE = 0.8
_MAX_MESSAGE_CHARS = 300  # earlier assistant replies can be long; the prompt budget is small


def _date(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_MESSAGE_CHARS else text[: _MAX_MESSAGE_CHARS - 1] + "…"


class MemoryManager:
    """Coordinates conversation state and persistent memory.

    Lifecycle:
        manager = MemoryManager(db_path)
        manager.start_session()
        ...
        manager.close()

    Retrieval is lexical (FTS5/BM25 blended with recency and importance).
    shortcut: no embeddings; measured as not worth it at this scale (see
    scripts/benchmark_memory.py notes), add them if paraphrase recall matters.
    """

    def __init__(self, store: SQLiteStore, *, cross_session: bool = True) -> None:
        self._store: SQLiteStore = store
        self._conversation: ConversationState | None = None
        # When on, retrieve_context also searches messages from earlier sessions.
        self.cross_session = cross_session

    @property
    def store(self) -> SQLiteStore:
        return self._store

    @property
    def conversation(self) -> ConversationState | None:
        return self._conversation

    def start_session(self) -> ConversationState:
        """Create a new conversation session."""
        session = ConversationSession.create()
        self._store.create_session(session)
        self._conversation = ConversationState(self._store, session)
        logger.info("SESSION_CREATED id=%s", session.id[:12])
        return self._conversation

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.NOTE,
        importance: float = 0.5,
        source_event_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Memory:
        """Store a persistent memory."""
        mem = Memory.create(
            content=content,
            memory_type=memory_type,
            importance=importance,
            source_event_id=source_event_id,
            metadata=metadata,
        )
        self._store.add(mem)
        logger.info(
            "MEMORY_WRITE id=%s type=%s importance=%.2f",
            mem.id[:12], mem.type.value, mem.importance,
        )
        return mem

    def forget(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._store.delete(memory_id)
        logger.info("MEMORY_DELETE id=%s", memory_id[:12])

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Search memories by keyword relevance + recency + importance."""
        results = self._store.search(query, limit=limit)
        logger.debug(
            "MEMORY_RETRIEVAL query=%r results=%d", query, len(results)
        )
        return results

    def retrieve_context(self, query: str, *, limit: int) -> list[str]:
        """Up to `limit` provenance-tagged lines relevant to `query`.

        Stored memories plus, when cross_session is on, messages from earlier
        sessions (never the current one, which the conversation history covers).
        Lines whose text differs only in case, punctuation or spacing are
        dropped (a stricter fuzzy match would merge facts differing in a number).
        """
        if limit <= 0:
            return []
        t0 = time.perf_counter()
        # (score, line, text); score is "lower is better" in both sources.
        scored = [
            (r[8], f"[memory, {_date(r[3])}] {r[2]}", r[2])
            for r in self._store.search_memory_rows(query, limit)
        ]
        if self.cross_session:
            current = self._conversation.session_id if self._conversation else None
            scored += [
                (score, f"[earlier conversation, {_date(ts)}, {role}] {_clip(content)}", content)
                for content, ts, role, score in self._store.search_message_rows(
                    query, limit, exclude_session_id=current
                )
            ]
        scored.sort(key=lambda x: x[0])
        lines: list[str] = []
        seen: set[str] = set()
        for _, line, text in scored:
            key = normalize(text)
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
            if len(lines) == limit:
                break
        logger.debug(
            "MEMORY_RETRIEVAL results=%d ms=%.1f", len(lines), (time.perf_counter() - t0) * 1000
        )
        return lines

    def maybe_remember(self, text: str, source_event_id: str) -> Memory | None:
        """Store an explicit "remember that ..." request; otherwise do nothing.

        Returns the stored memory (the existing one when the same fact was
        already stored), or None when the text is not such a request.
        """
        utterance = text.strip().replace("\u2019", "'")
        if utterance.endswith("?"):
            return None
        utterance = _LEAD.sub("", utterance).rstrip(" .!")
        for pattern, memory_type, template in _REMEMBER_RULES:
            m = pattern.match(utterance)
            if m is None:
                continue
            fact = m.group("fact").strip(" ,.")
            if len(fact.split()) < 2 or _NOT_A_FACT.match(fact):
                return None
            mem, created = self._store.add_unique(Memory.create(
                content=template.format(fact),
                memory_type=memory_type,
                importance=_EXPLICIT_IMPORTANCE,
                source_event_id=source_event_id,
                metadata={
                    "session_id": self._conversation.session_id if self._conversation else None,
                    "origin": "explicit_request",
                },
            ))
            logger.info(
                "MEMORY_WRITE id=%s type=%s explicit=1 created=%s",
                mem.id[:12], mem.type.value, created,
            )
            return mem
        return None

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._store.get(memory_id)

    def clear_session(self, session_id: str) -> None:
        """Delete all messages and session record for a session."""
        self._store.clear_session(session_id)
        logger.info("SESSION_CLEARED id=%s", session_id[:12])

    def close(self) -> None:
        self._store.close()
