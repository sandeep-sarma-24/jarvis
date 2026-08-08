"""SQLite-backed memory and conversation store.

Local, persistent, transactional. No external services.

Thread safety: each worker thread gets its own sqlite3.Connection via thread-local
storage. All connections are registered for lifecycle management.

Connection lifecycle: uses a Condition-based state machine (OPEN → CLOSING → CLOSED).
Operation admission and close are atomic under the same lock, eliminating races.
close() waits for active operations to drain, then closes all connections.
check_same_thread=False permits this controlled cross-thread shutdown.

In-memory behavior: for ":memory:" databases, a shared-cache URI is used so all
threads see the same in-memory database. A keeper connection holds the database
alive for the store's lifetime.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
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

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_source_event_unique
    ON conversation_messages(source_event_id) WHERE source_event_id IS NOT NULL;

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
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


def _configure_connection(conn: sqlite3.Connection, is_file: bool) -> None:
    """Apply WAL (file only), foreign keys, and busy timeout to a connection."""
    if is_file:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")


class SQLiteStore:
    """Unified SQLite store implementing both MemoryStore and ConversationStore.

    State machine: OPEN → CLOSING → CLOSED.

    Operation admission and close() are serialized under a single Condition,
    eliminating the race between checking closed state and incrementing the
    active operation counter.

    Each thread gets its own sqlite3.Connection (thread-local). All connections
    use check_same_thread=False so close() can shut them down from any thread.
    This does NOT mean connections are shared for operations — each thread uses
    only its own connection. The flag exists solely for controlled lifecycle.

    ponytail: condition + active-op counter, no pool. Single-user local.
    Upgrade path: connection pool if dozens of concurrent threads are needed.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._raw_path = str(db_path)
        self._is_file = self._raw_path != ":memory:"

        if self._is_file:
            self._connect_uri = None
            self._keeper: sqlite3.Connection | None = None
        else:
            name = uuid.uuid4().hex
            self._connect_uri = f"file:{name}?mode=memory&cache=shared"
            self._keeper = sqlite3.connect(
                self._connect_uri, uri=True, check_same_thread=False
            )

        self._local = threading.local()
        self._conn_lock = threading.Lock()
        self._connections: set[sqlite3.Connection] = set()

        # Lifecycle state: protected by _lifecycle_cond
        self._lifecycle_cond = threading.Condition()
        self._active_ops = 0
        self._closing = False
        self._closed = False

        conn = self._get_conn()
        conn.executescript(_SCHEMA)

    def _make_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection for the current configuration."""
        if self._connect_uri:
            conn = sqlite3.connect(
                self._connect_uri, uri=True,
                isolation_level="DEFERRED", check_same_thread=False,
            )
        else:
            conn = sqlite3.connect(
                self._raw_path, isolation_level="DEFERRED", check_same_thread=False,
            )
        _configure_connection(conn, self._is_file)
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local connection, registered for lifecycle.

        Refuses only after CLOSED (not during CLOSING — already-admitted ops
        must still access their connection).
        """
        if self._closed:
            raise RuntimeError("SQLiteStore is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_connection()
            self._local.conn = conn
            with self._conn_lock:
                self._connections.add(conn)
        return conn

    def _begin_op(self) -> None:
        """Atomically admit an operation. Raises if store is closing/closed."""
        with self._lifecycle_cond:
            if self._closing or self._closed:
                raise RuntimeError("SQLiteStore is closed")
            self._active_ops += 1

    def _end_op(self) -> None:
        """Mark an operation complete. Notifies close() if draining."""
        with self._lifecycle_cond:
            self._active_ops -= 1
            if self._active_ops == 0:
                self._lifecycle_cond.notify_all()

    def close(self) -> None:
        """Transition OPEN → CLOSING → CLOSED. Waits for active ops to drain.

        Safe to call from any thread. Idempotent.
        """
        with self._lifecycle_cond:
            if self._closed:
                return
            if self._closing:
                # Another thread is already closing; wait for completion
                self._lifecycle_cond.wait_for(lambda: self._closed)
                return
            self._closing = True
            # Wait for all active operations to finish
            self._lifecycle_cond.wait_for(lambda: self._active_ops == 0)
            self._closed = True

        # Now no new ops can be admitted and none are in-flight
        with self._conn_lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
        if self._keeper is not None:
            self._keeper.close()
            self._keeper = None

    # ── MemoryStore ──────────────────────────────────────────────────────

    def add(self, memory: Memory) -> None:
        self._begin_op()
        try:
            conn = self._get_conn()
            conn.execute(
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
            conn.commit()
        finally:
            self._end_op()

    def get(self, memory_id: str) -> Memory | None:
        self._begin_op()
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT id, type, content, created_at, updated_at, importance, source_event_id, metadata_json FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_memory(row)
        finally:
            self._end_op()

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

        self._begin_op()
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT id, type, content, created_at, updated_at, importance, source_event_id, metadata_json FROM memories ORDER BY created_at DESC LIMIT 1000"
            ).fetchall()
        finally:
            self._end_op()

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
        self._begin_op()
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
        finally:
            self._end_op()

    def all_memories(self, limit: int = 100) -> list[Memory]:
        self._begin_op()
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT id, type, content, created_at, updated_at, importance, source_event_id, metadata_json FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            self._end_op()

    # ── ConversationStore ────────────────────────────────────────────────

    def create_session(self, session: ConversationSession) -> None:
        self._begin_op()
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR IGNORE INTO conversation_sessions (id, started_at, last_activity_at) VALUES (?, ?, ?)",
                (session.id, session.started_at, session.last_activity_at),
            )
            conn.commit()
        finally:
            self._end_op()

    def get_session(self, session_id: str) -> ConversationSession | None:
        self._begin_op()
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT id, started_at, last_activity_at FROM conversation_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return ConversationSession(id=row[0], started_at=row[1], last_activity_at=row[2])
        finally:
            self._end_op()

    def update_session_activity(self, session_id: str, timestamp: float) -> None:
        self._begin_op()
        try:
            conn = self._get_conn()
            conn.execute(
                "UPDATE conversation_sessions SET last_activity_at = ? WHERE id = ?",
                (timestamp, session_id),
            )
            conn.commit()
        finally:
            self._end_op()

    def add_message(self, message: ConversationMessage) -> None:
        self._begin_op()
        try:
            conn = self._get_conn()
            if message.source_event_id:
                try:
                    conn.execute(
                        "INSERT INTO conversation_messages (id, session_id, role, content, timestamp, source_event_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            message.id,
                            message.session_id,
                            message.role,
                            message.content,
                            message.timestamp,
                            message.source_event_id,
                        ),
                    )
                except sqlite3.IntegrityError:
                    return
                except sqlite3.OperationalError:
                    existing = self._get_message_by_source_inner(message.source_event_id)
                    if existing is not None:
                        return
                    raise
            else:
                conn.execute(
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
            conn.commit()
        finally:
            self._end_op()
        self.update_session_activity(message.session_id, message.timestamp)

    def get_messages(
        self, session_id: str, limit: int = 50
    ) -> list[ConversationMessage]:
        self._begin_op()
        try:
            conn = self._get_conn()
            rows = conn.execute(
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
        finally:
            self._end_op()

    def get_message_by_source(self, source_event_id: str) -> ConversationMessage | None:
        self._begin_op()
        try:
            return self._get_message_by_source_inner(source_event_id)
        finally:
            self._end_op()

    def _get_message_by_source_inner(self, source_event_id: str) -> ConversationMessage | None:
        """Inner implementation without op guard (for use within add_message)."""
        conn = self._get_conn()
        row = conn.execute(
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
        self._begin_op()
        try:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM conversation_messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM conversation_sessions WHERE id = ?", (session_id,)
            )
            conn.commit()
        finally:
            self._end_op()

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
