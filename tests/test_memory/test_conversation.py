"""Tests for conversation state management."""
import time

import pytest

from app.memory.conversation import ConversationState
from app.memory.sqlite_store import SQLiteStore
from app.memory.types import ConversationSession


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def conversation(store):
    session = ConversationSession.create()
    store.create_session(session)
    return ConversationState(store, session)


class TestConversationMessages:

    def test_add_user_message(self, conversation):
        msg = conversation.add_user_message("Hello Jarvis")
        assert msg.role == "user"
        assert msg.content == "Hello Jarvis"
        assert msg.session_id == conversation.session_id

    def test_add_assistant_message(self, conversation):
        msg = conversation.add_assistant_message("Hello! How can I help?")
        assert msg.role == "assistant"
        assert msg.content == "Hello! How can I help?"

    def test_recent_messages_order(self, conversation):
        conversation.add_user_message("first")
        conversation.add_assistant_message("response one")
        conversation.add_user_message("second")
        conversation.add_assistant_message("response two")

        messages = conversation.recent_messages(limit=10)
        assert len(messages) == 4
        assert messages[0].content == "first"
        assert messages[1].content == "response one"
        assert messages[2].content == "second"
        assert messages[3].content == "response two"

    def test_recent_messages_roles_correct(self, conversation):
        conversation.add_user_message("user msg")
        conversation.add_assistant_message("assistant msg")

        messages = conversation.recent_messages()
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_recent_messages_limit(self, conversation):
        for i in range(20):
            conversation.add_user_message(f"msg {i}")
        messages = conversation.recent_messages(limit=5)
        assert len(messages) == 5

    def test_source_event_id_preserved(self, conversation):
        msg = conversation.add_user_message("test", source_event_id="evt_abc")
        messages = conversation.recent_messages()
        assert messages[0].source_event_id == "evt_abc"


class TestIdempotency:

    def test_duplicate_source_event_id(self, conversation):
        """Same source_event_id should not create duplicate messages."""
        conversation.add_user_message("first", source_event_id="evt_1")
        conversation.add_user_message("second attempt", source_event_id="evt_1")

        messages = conversation.recent_messages()
        assert len(messages) == 1
        assert messages[0].content == "first"

    def test_different_source_event_ids(self, conversation):
        conversation.add_user_message("first", source_event_id="evt_1")
        conversation.add_user_message("second", source_event_id="evt_2")

        messages = conversation.recent_messages()
        assert len(messages) == 2

    def test_no_source_event_id_allows_multiple(self, conversation):
        """Messages without source_event_id are always added."""
        conversation.add_user_message("first")
        conversation.add_user_message("second")
        messages = conversation.recent_messages()
        assert len(messages) == 2


class TestHasMessageForEvent:

    def test_returns_true_when_exists(self, conversation):
        conversation.add_user_message("test", source_event_id="evt_x")
        assert conversation.has_message_for_event("evt_x") is True

    def test_returns_false_when_not_exists(self, conversation):
        assert conversation.has_message_for_event("nonexistent") is False


class TestSessionActivity:

    def test_session_activity_updated_on_message(self, store, conversation):
        original = conversation.session.last_activity_at
        time.sleep(0.01)
        conversation.add_user_message("update activity")

        updated_session = store.get_session(conversation.session_id)
        assert updated_session.last_activity_at > original
