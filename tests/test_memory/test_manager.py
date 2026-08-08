"""Tests for MemoryManager."""
import pytest

from app.memory.manager import MemoryManager
from app.memory.sqlite_store import SQLiteStore
from app.memory.types import MemoryType


@pytest.fixture
def manager():
    store = SQLiteStore(":memory:")
    m = MemoryManager(store)
    yield m
    m.close()


class TestSessionLifecycle:

    def test_start_session_creates_conversation(self, manager):
        conv = manager.start_session()
        assert conv is not None
        assert conv.session_id is not None
        assert manager.conversation is conv

    def test_multiple_start_session_creates_new(self, manager):
        conv1 = manager.start_session()
        conv2 = manager.start_session()
        assert conv1.session_id != conv2.session_id


class TestRemember:

    def test_remember_creates_memory(self, manager):
        manager.start_session()
        mem = manager.remember("User prefers Neovim", memory_type=MemoryType.PREFERENCE, importance=0.8)
        assert mem.content == "User prefers Neovim"
        assert mem.type == MemoryType.PREFERENCE
        assert abs(mem.importance - 0.8) < 0.01

    def test_remember_retrievable(self, manager):
        manager.start_session()
        mem = manager.remember("important fact")
        retrieved = manager.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.content == "important fact"


class TestForget:

    def test_forget_removes_memory(self, manager):
        manager.start_session()
        mem = manager.remember("to forget")
        manager.forget(mem.id)
        assert manager.get_memory(mem.id) is None

    def test_forget_nonexistent_no_error(self, manager):
        manager.forget("nonexistent")


class TestSearch:

    def test_search_finds_relevant(self, manager):
        manager.start_session()
        manager.remember("User prefers Neovim editor", memory_type=MemoryType.PREFERENCE)
        manager.remember("User likes pizza", memory_type=MemoryType.PREFERENCE)

        results = manager.search("preferred editor")
        assert any("Neovim" in r.content for r in results)

    def test_search_respects_limit(self, manager):
        manager.start_session()
        for i in range(20):
            manager.remember(f"fact about topic {i}")
        results = manager.search("topic", limit=3)
        assert len(results) <= 3


class TestClearSession:

    def test_clear_session_removes_messages(self, manager):
        conv = manager.start_session()
        conv.add_user_message("hello")
        conv.add_assistant_message("hi")

        manager.clear_session(conv.session_id)

        store = manager.store
        messages = store.get_messages(conv.session_id)
        assert len(messages) == 0

    def test_clear_session_removes_session(self, manager):
        conv = manager.start_session()
        session_id = conv.session_id
        manager.clear_session(session_id)

        assert manager.store.get_session(session_id) is None
