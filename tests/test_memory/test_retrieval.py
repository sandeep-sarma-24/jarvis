"""Tests for memory retrieval scoring."""
import time

import pytest

from app.memory.sqlite_store import SQLiteStore
from app.memory.types import Memory, MemoryType


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


class TestRetrievalScoring:
    """Verify deterministic keyword + recency + importance scoring."""

    def test_exact_keyword_match_ranked_high(self, store):
        store.add(Memory.create(content="User prefers Neovim editor", memory_type=MemoryType.PREFERENCE, importance=0.8))
        store.add(Memory.create(content="User likes pizza", memory_type=MemoryType.PREFERENCE, importance=0.8))

        results = store.search("preferred editor", limit=5)
        assert len(results) >= 1
        assert "Neovim" in results[0].content or "editor" in results[0].content

    def test_irrelevant_memory_not_returned(self, store):
        """Pizza memory should not appear for database queries."""
        store.add(Memory.create(content="User likes pizza", memory_type=MemoryType.PREFERENCE, importance=0.9))
        results = store.search("database connection", limit=5)
        assert not any("pizza" in r.content for r in results)

    def test_importance_weight(self, store):
        store.add(Memory.create(content="low priority python info", memory_type=MemoryType.NOTE, importance=0.1))
        store.add(Memory.create(content="high priority python config", memory_type=MemoryType.NOTE, importance=0.95))

        results = store.search("python", limit=2)
        assert len(results) == 2
        assert results[0].importance > results[1].importance

    def test_multiple_keyword_match_scores_higher(self, store):
        store.add(Memory.create(content="database migration", memory_type=MemoryType.FACT, importance=0.5))
        store.add(Memory.create(content="database connection pool timeout config", memory_type=MemoryType.FACT, importance=0.5))

        results = store.search("database connection timeout", limit=2)
        assert len(results) == 2
        assert "connection" in results[0].content and "timeout" in results[0].content

    def test_single_word_queries(self, store):
        store.add(Memory.create(content="Python is the preferred language", memory_type=MemoryType.PREFERENCE))
        results = store.search("Python", limit=5)
        assert len(results) == 1

    def test_case_insensitive_matching(self, store):
        store.add(Memory.create(content="Neovim is the preferred editor", memory_type=MemoryType.PREFERENCE))
        results = store.search("neovim", limit=5)
        assert len(results) == 1
        results = store.search("NEOVIM", limit=5)
        assert len(results) == 1
