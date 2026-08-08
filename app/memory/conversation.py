"""Conversation state — manages session and messages for LLM context."""
from __future__ import annotations

import logging
from typing import Literal

from app.memory.store import ConversationStore
from app.memory.types import ConversationMessage, ConversationSession

logger = logging.getLogger(__name__)


class ConversationState:
    """Active conversation state for the current session.

    Provides bounded access to recent messages for context building.
    Persists through the ConversationStore protocol.
    """

    def __init__(self, store: ConversationStore, session: ConversationSession) -> None:
        self._store = store
        self._session = session

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def session(self) -> ConversationSession:
        return self._session

    def add_user_message(
        self, content: str, source_event_id: str | None = None
    ) -> ConversationMessage:
        msg = ConversationMessage.create(
            session_id=self._session.id,
            role="user",
            content=content,
            source_event_id=source_event_id,
        )
        self._store.add_message(msg)
        logger.debug(
            "CONVERSATION_MESSAGE session=%s role=user len=%d source=%s",
            self._session.id[:12], len(content), source_event_id[:12] if source_event_id else None,
        )
        return msg

    def add_assistant_message(
        self, content: str, source_event_id: str | None = None
    ) -> ConversationMessage:
        msg = ConversationMessage.create(
            session_id=self._session.id,
            role="assistant",
            content=content,
            source_event_id=source_event_id,
        )
        self._store.add_message(msg)
        logger.debug(
            "CONVERSATION_MESSAGE session=%s role=assistant len=%d source=%s",
            self._session.id[:12], len(content), source_event_id[:12] if source_event_id else None,
        )
        return msg

    def recent_messages(self, limit: int = 20) -> list[ConversationMessage]:
        return self._store.get_messages(self._session.id, limit=limit)

    def has_message_for_event(self, source_event_id: str) -> bool:
        return self._store.get_message_by_source(source_event_id) is not None
