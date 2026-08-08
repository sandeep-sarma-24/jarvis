"""Memory Manager — coordinates conversation, persistent memory, and retrieval."""
from __future__ import annotations

import logging
import time

from app.memory.conversation import ConversationState
from app.memory.sqlite_store import SQLiteStore
from app.memory.store import ConversationStore, MemoryStore
from app.memory.types import ConversationSession, Memory, MemoryType

logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinates conversation state and persistent memory.

    Lifecycle:
        manager = MemoryManager(db_path)
        manager.start_session()
        ...
        manager.close()

    ponytail: retrieval is keyword + recency + importance. No embeddings.
    Upgrade path: add semantic retrieval via embeddings in a later phase.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self._store: SQLiteStore = store
        self._conversation: ConversationState | None = None

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

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._store.get(memory_id)

    def clear_session(self, session_id: str) -> None:
        """Delete all messages and session record for a session."""
        self._store.clear_session(session_id)
        logger.info("SESSION_CLEARED id=%s", session_id[:12])

    def close(self) -> None:
        self._store.close()
