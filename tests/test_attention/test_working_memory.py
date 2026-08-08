"""Tests for bounded working memory."""
import time

import pytest

from app.memory.working import WorkingMemory


@pytest.fixture
def mem():
    return WorkingMemory(max_size=5)


class TestBounded:

    def test_bounded_size(self, mem):
        for i in range(10):
            mem.record(f"e{i}", f"text {i}", should_reason=False)
        assert len(mem) == 5

    def test_newest_retained(self, mem):
        for i in range(10):
            mem.record(f"e{i}", f"text {i}", should_reason=False)
        ids = [e.event_id for e in mem.entries]
        assert ids == [f"e{i}" for i in range(5, 10)]

    def test_old_evicted(self, mem):
        for i in range(10):
            mem.record(f"e{i}", f"text {i}", should_reason=False)
        ids = [e.event_id for e in mem.entries]
        assert "e0" not in ids

    def test_max_size_property(self, mem):
        assert mem.max_size == 5


class TestInteractionTracking:

    def test_last_interaction_none_initially(self, mem):
        assert mem.last_interaction_time is None
        assert mem.seconds_since_last_interaction() is None

    def test_last_interaction_updated_on_reason(self, mem):
        mem.record("e1", "hey jarvis", should_reason=True)
        assert mem.last_interaction_time is not None
        elapsed = mem.seconds_since_last_interaction()
        assert elapsed is not None
        assert elapsed < 1.0

    def test_last_interaction_not_updated_on_no_reason(self, mem):
        mem.record("e1", "background", should_reason=False)
        assert mem.last_interaction_time is None

    def test_last_should_reason(self, mem):
        assert mem.last_should_reason is None
        mem.record("e1", "test", should_reason=True)
        assert mem.last_should_reason is True
        mem.record("e2", "bg", should_reason=False)
        assert mem.last_should_reason is False


class TestClear:

    def test_clear_empties(self, mem):
        mem.record("e1", "test", should_reason=True)
        mem.clear()
        assert len(mem) == 0
        assert mem.last_interaction_time is None
        assert mem.last_should_reason is None


class TestNoPersistence:

    def test_no_disk_access(self, mem, tmp_path):
        """Working memory must not write to disk."""
        import os
        before = set(os.listdir(tmp_path))
        for i in range(20):
            mem.record(f"e{i}", f"text {i}", should_reason=i % 3 == 0)
        after = set(os.listdir(tmp_path))
        assert before == after


class TestEntryShape:

    def test_entry_has_fields(self, mem):
        mem.record("e1", "hello world", should_reason=True)
        e = mem.entries[0]
        assert e.event_id == "e1"
        assert e.transcript == "hello world"
        assert e.should_reason is True
        assert e.timestamp > 0
