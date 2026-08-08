"""SQLite-backed memory and conversation store.

Local, persistent, transactional. No external services.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from app.memory.store import ConversationStore, MemoryStore
from app.memory.types import ConversationMessage, ConversationSession, Memory, MemoryType

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    last_activity_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    source_event_id TEXT,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(id)
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    source_event_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON conversation_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON conversation_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_source_event ON conversation_messages(source_event_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


class SQLiteStore:
    """Unified SQLite store implementing both MemoryStore and ConversationStore.

    ponytail: single-connection, synchronous. Caller wraps in asyncio.to_thread.
    Upgrade path: connection pool if concurrent writes become a bottleneck.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, isolation_level="DEFERRED")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ── MemoryStore ──────────────────────────────────────────────────────

    def add(self, memory: Memory) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO memories (id, type, content, created_at, updated_at, importance, source_event_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.id,
                memory.type.value,
                memory.content,
                memory.created_at,
                memory.updated_at,
                memory.importance,
                memory.source_event_id,
                json.dumps(memory.metadata),
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT id, type, content, created_at, updated_at, importance, source_event_id, metadata_json FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Keyword + recency + importance scoring.

        Score = lexical_match + recency_weight + importance_weight.
        Lexical match: fraction of query words found in content (case-insensitive).
        Recency: decays over 7 days.
        Importance: direct weight.
        """
        words = [w.lower() for w in query.split() if len(w) > 1]
        if not words:
            return []

        rows = self._conn.execute(
            "SELECT id, type, content, created_at, updated_at, importance, source_event_id, metadata_json FROM memories ORDER BY created_at DESC LIMIT 1000"
        ).fetchall()

        now = time.time()
        scored: list[tuple[float, Memory]] = []

        for row in rows:
            mem = self._row_to_memory(row)
            content_lower = mem.content.lower()

            match_count = sum(1 for w in words if w in content_lower)
            lexical_score = match_count / len(words) if words else 0.0

            age_days = (now - mem.created_at) / 86400
            recency_score = max(0.0, 1.0 - (age_days / 7.0))

            score = (lexical_score * 0.6) + (recency_score * 0.2) + (mem.importance * 0.2)

            if lexical_score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    def delete(self, memory_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def all_memories(self, limit: int = 100) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT id, type, content, created_at, updated_at, importance, source_event_id, metadata_json FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ── ConversationStore ────────────────────────────────────────────────

    def create_session(self, session: ConversationSession) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO conversation_sessions (id, started_at, last_activity_at) VALUES (?, ?, ?)",
            (session.id, session.started_at, session.last_activity_at),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> ConversationSession | None:
        row = self._conn.execute(
            "SELECT id, started_at, last_activity_at FROM conversation_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return ConversationSession(id=row[0], started_at=row[1], last_activity_at=row[2])

    def update_session_activity(self, session_id: str, timestamp: float) -> None:
        self._conn.execute(
            "UPDATE conversation_sessions SET last_activity_at = ? WHERE id = ?",
            (timestamp, session_id),
        )
        self._conn.commit()

    def add_message(self, message: ConversationMessage) -> None:
        # Idempotency: if source_event_id already exists, skip
        if message.source_event_id:
            existing = self.get_message_by_source(message.source_event_id)
            if existing is not None:
                return
        self._conn.execute(
            "INSERT OR IGNORE INTO conversation_messages (id, session_id, role, content, timestamp, source_event_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                message.id,
                message.session_id,
                message.role,
                message.content,
                message.timestamp,
                message.source_event_id,
            ),
        )
        self._conn.commit()
        self.update_session_activity(message.session_id, message.timestamp)

    def get_messages(
        self, session_id: str, limit: int = 50
    ) -> list[ConversationMessage]:
        rows = self._conn.execute(
            "SELECT id, session_id, role, content, timestamp, source_event_id FROM conversation_messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        msgs = [
            ConversationMessage(
                id=r[0], session_id=r[1], role=r[2],
                content=r[3], timestamp=r[4], source_event_id=r[5],
            )
            for r in rows
        ]
        msgs.reverse()
        return msgs

    def get_message_by_source(self, source_event_id: str) -> ConversationMessage | None:
        row = self._conn.execute(
            "SELECT id, session_id, role, content, timestamp, source_event_id FROM conversation_messages WHERE source_event_id = ?",
            (source_event_id,),
        ).fetchone()
        if row is None:
            return None
        return ConversationMessage(
            id=row[0], session_id=row[1], role=row[2],
            content=row[3], timestamp=row[4], source_event_id=row[5],
        )

    def clear_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM conversation_messages WHERE session_id = ?", (session_id,)
        )
        self._conn.execute(
            "DELETE FROM conversation_sessions WHERE id = ?", (session_id,)
        )
        self._conn.commit()

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_memory(row: tuple) -> Memory:
        return Memory(
            id=row[0],
            type=MemoryType(row[1]),
            content=row[2],
            created_at=row[3],
            updated_at=row[4],
            importance=row[5],
            source_event_id=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
        )
