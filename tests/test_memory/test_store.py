"""Tests for memory store (SQLite implementation)."""
import time

import pytest

from app.memory.sqlite_store import SQLiteStore
from app.memory.types import Memory, MemoryType


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


class TestMemoryAdd:

    def test_add_and_get(self, store):
        mem = Memory.create(content="User prefers Neovim", memory_type=MemoryType.PREFERENCE)
        store.add(mem)
        retrieved = store.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "User prefers Neovim"
        assert retrieved.type == MemoryType.PREFERENCE

    def test_add_duplicate_is_idempotent(self, store):
        mem = Memory.create(content="test", memory_type=MemoryType.NOTE)
        store.add(mem)
        store.add(mem)  # no error
        assert store.get(mem.id) is not None

    def test_importance_preserved(self, store):
        mem = Memory.create(content="important", memory_type=MemoryType.FACT, importance=0.9)
        store.add(mem)
        retrieved = store.get(mem.id)
        assert abs(retrieved.importance - 0.9) < 0.01

    def test_metadata_preserved(self, store):
        mem = Memory.create(
            content="has meta", memory_type=MemoryType.NOTE,
            metadata={"key": "value", "num": 42},
        )
        store.add(mem)
        retrieved = store.get(mem.id)
        assert retrieved.metadata["key"] == "value"
        assert retrieved.metadata["num"] == 42

    def test_source_event_id_preserved(self, store):
        mem = Memory.create(
            content="linked", memory_type=MemoryType.CONVERSATION,
            source_event_id="evt123",
        )
        store.add(mem)
        retrieved = store.get(mem.id)
        assert retrieved.source_event_id == "evt123"


class TestMemoryGet:

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent") is None


class TestMemoryDelete:

    def test_delete_removes_memory(self, store):
        mem = Memory.create(content="to delete", memory_type=MemoryType.NOTE)
        store.add(mem)
        store.delete(mem.id)
        assert store.get(mem.id) is None

    def test_delete_nonexistent_no_error(self, store):
        store.delete("nonexistent")  # no error


class TestMemorySearch:

    def test_search_by_keyword(self, store):
        store.add(Memory.create(content="User prefers Neovim editor", memory_type=MemoryType.PREFERENCE))
        store.add(Memory.create(content="User likes pizza", memory_type=MemoryType.PREFERENCE))
        store.add(Memory.create(content="Database connection string", memory_type=MemoryType.FACT))

        results = store.search("preferred editor", limit=5)
        assert len(results) >= 1
        assert any("Neovim" in r.content for r in results)

    def test_search_relevance_ordering(self, store):
        store.add(Memory.create(content="User prefers Neovim editor", memory_type=MemoryType.PREFERENCE, importance=0.9))
        store.add(Memory.create(content="pizza delivery phone number", memory_type=MemoryType.NOTE, importance=0.3))

        results = store.search("editor", limit=5)
        if len(results) >= 1:
            assert "Neovim" in results[0].content

    def test_search_empty_query_returns_empty(self, store):
        store.add(Memory.create(content="something", memory_type=MemoryType.NOTE))
        results = store.search("", limit=5)
        assert results == []

    def test_search_no_match_returns_empty(self, store):
        store.add(Memory.create(content="cats and dogs", memory_type=MemoryType.NOTE))
        results = store.search("quantum physics database", limit=5)
        assert results == []

    def test_search_limit_respected(self, store):
        for i in range(20):
            store.add(Memory.create(content=f"memory about topic {i}", memory_type=MemoryType.NOTE))
        results = store.search("topic", limit=3)
        assert len(results) <= 3

    def test_importance_affects_ranking(self, store):
        store.add(Memory.create(content="low importance python", memory_type=MemoryType.NOTE, importance=0.1))
        store.add(Memory.create(content="high importance python", memory_type=MemoryType.NOTE, importance=0.9))
        results = store.search("python", limit=2)
        assert len(results) == 2
        assert results[0].importance >= results[1].importance

    def test_irrelevant_memory_excluded(self, store):
        """Memory about pizza should not dominate results for database queries."""
        store.add(Memory.create(content="User likes pizza", memory_type=MemoryType.PREFERENCE, importance=0.9))
        store.add(Memory.create(content="database connection timeout is 30s", memory_type=MemoryType.FACT, importance=0.5))

        results = store.search("database connection", limit=5)
        if results:
            assert "database" in results[0].content.lower()


class TestAllMemories:

    def test_all_memories_returns_recent_first(self, store):
        m1 = Memory.create(content="first", memory_type=MemoryType.NOTE)
        store.add(m1)
        time.sleep(0.01)
        m2 = Memory.create(content="second", memory_type=MemoryType.NOTE)
        store.add(m2)
        all_mems = store.all_memories(limit=10)
        assert len(all_mems) == 2
        assert all_mems[0].content == "second"

    def test_all_memories_limit(self, store):
        for i in range(10):
            store.add(Memory.create(content=f"mem {i}", memory_type=MemoryType.NOTE))
        assert len(store.all_memories(limit=3)) == 3
