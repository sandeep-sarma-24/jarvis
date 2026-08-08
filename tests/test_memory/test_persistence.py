"""Tests for memory persistence — verifies data survives database reopen."""
import tempfile
import time
from pathlib import Path

import pytest

from app.memory.manager import MemoryManager
from app.memory.sqlite_store import SQLiteStore
from app.memory.types import MemoryType


class TestPersistenceAcrossRestart:
    """Simulate process restart by closing and reopening the database."""

    def test_memory_survives_restart(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Process 1: create and store
        store1 = SQLiteStore(db_path)
        manager1 = MemoryManager(store1)
        manager1.start_session()
        mem = manager1.remember("User prefers Neovim", memory_type=MemoryType.PREFERENCE, importance=0.8)
        mem_id = mem.id
        manager1.close()

        # Process 2: reopen and retrieve
        store2 = SQLiteStore(db_path)
        manager2 = MemoryManager(store2)
        retrieved = manager2.get_memory(mem_id)
        assert retrieved is not None
        assert retrieved.content == "User prefers Neovim"
        assert retrieved.type == MemoryType.PREFERENCE
        assert abs(retrieved.importance - 0.8) < 0.01
        manager2.close()

    def test_conversation_survives_restart(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Process 1: create session and messages
        store1 = SQLiteStore(db_path)
        manager1 = MemoryManager(store1)
        conv1 = manager1.start_session()
        session_id = conv1.session_id
        conv1.add_user_message("Hello Jarvis", source_event_id="evt1")
        conv1.add_assistant_message("Hello! How can I help?", source_event_id="evt1_resp")
        manager1.close()

        # Process 2: reopen and verify
        store2 = SQLiteStore(db_path)
        messages = store2.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello Jarvis"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hello! How can I help?"
        store2.close()

    def test_session_survives_restart(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Process 1
        store1 = SQLiteStore(db_path)
        manager1 = MemoryManager(store1)
        conv1 = manager1.start_session()
        session_id = conv1.session_id
        manager1.close()

        # Process 2
        store2 = SQLiteStore(db_path)
        session = store2.get_session(session_id)
        assert session is not None
        assert session.id == session_id
        store2.close()

    def test_multiple_sessions_persist(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Process 1: create two sessions
        store1 = SQLiteStore(db_path)
        manager1 = MemoryManager(store1)
        conv1 = manager1.start_session()
        sid1 = conv1.session_id
        conv1.add_user_message("session 1 message")
        conv2 = manager1.start_session()
        sid2 = conv2.session_id
        conv2.add_user_message("session 2 message")
        manager1.close()

        # Process 2: verify both exist
        store2 = SQLiteStore(db_path)
        assert store2.get_session(sid1) is not None
        assert store2.get_session(sid2) is not None
        msgs1 = store2.get_messages(sid1)
        msgs2 = store2.get_messages(sid2)
        assert len(msgs1) == 1
        assert msgs1[0].content == "session 1 message"
        assert len(msgs2) == 1
        assert msgs2[0].content == "session 2 message"
        store2.close()

    def test_search_works_after_restart(self, tmp_path):
        db_path = tmp_path / "test.db"

        store1 = SQLiteStore(db_path)
        manager1 = MemoryManager(store1)
        manager1.remember("Python is the preferred language", memory_type=MemoryType.PREFERENCE)
        manager1.close()

        store2 = SQLiteStore(db_path)
        manager2 = MemoryManager(store2)
        results = manager2.search("Python language")
        assert len(results) >= 1
        assert "Python" in results[0].content
        manager2.close()

    def test_delete_persists(self, tmp_path):
        db_path = tmp_path / "test.db"

        store1 = SQLiteStore(db_path)
        manager1 = MemoryManager(store1)
        mem = manager1.remember("to delete")
        mem_id = mem.id
        manager1.forget(mem_id)
        manager1.close()

        store2 = SQLiteStore(db_path)
        manager2 = MemoryManager(store2)
        assert manager2.get_memory(mem_id) is None
        manager2.close()


class TestNewSessionOnStartup:
    """Each application start creates a new session while retaining history."""

    def test_new_session_each_start(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Start 1
        store1 = SQLiteStore(db_path)
        m1 = MemoryManager(store1)
        c1 = m1.start_session()
        sid1 = c1.session_id
        c1.add_user_message("first session")
        m1.close()

        # Start 2
        store2 = SQLiteStore(db_path)
        m2 = MemoryManager(store2)
        c2 = m2.start_session()
        sid2 = c2.session_id
        m2.close()

        assert sid1 != sid2

        # Both sessions still in DB
        store3 = SQLiteStore(db_path)
        assert store3.get_session(sid1) is not None
        assert store3.get_session(sid2) is not None
        msgs = store3.get_messages(sid1)
        assert len(msgs) == 1
        store3.close()
