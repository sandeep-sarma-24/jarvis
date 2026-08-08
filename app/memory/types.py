"""Memory type definitions."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class MemoryType(Enum):
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    NOTE = "note"


@dataclass(frozen=True)
class Memory:
    id: str
    type: MemoryType
    content: str
    created_at: float
    updated_at: float
    importance: float
    source_event_id: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def create(
        *,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        source_event_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Memory:
        now = time.time()
        return Memory(
            id=uuid.uuid4().hex,
            type=memory_type,
            content=content,
            created_at=now,
            updated_at=now,
            importance=max(0.0, min(1.0, importance)),
            source_event_id=source_event_id,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ConversationMessage:
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: float
    source_event_id: str | None = None

    @staticmethod
    def create(
        *,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        source_event_id: str | None = None,
    ) -> ConversationMessage:
        return ConversationMessage(
            id=uuid.uuid4().hex,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=time.time(),
            source_event_id=source_event_id,
        )


@dataclass(frozen=True)
class ConversationSession:
    id: str
    started_at: float
    last_activity_at: float

    @staticmethod
    def create() -> ConversationSession:
        now = time.time()
        return ConversationSession(
            id=uuid.uuid4().hex,
            started_at=now,
            last_activity_at=now,
        )
