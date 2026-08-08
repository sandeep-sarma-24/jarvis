"""Tests for SQLite thread safety and async concurrency.

Verifies that database operations work correctly when called through
asyncio.to_thread() from multiple worker threads simultaneously.
"""
import asyncio

import pytest

from app.memory.manager import MemoryManager
from app.memory.sqlite_store import SQLiteStore
from app.memory.types import ConversationSession, MemoryType


@pytest.fixture
def store(tmp_path):
    """File-backed store (not :memory:) to test real multi-thread WAL behavior."""
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def manager(store):
    m = MemoryManager(store)
    m.start_session()
    return m


class TestAsyncToThread:
    """Verify operations work through asyncio.to_thread — the actual app pattern."""

    async def test_remember_via_to_thread(self, manager):
        mem = await asyncio.to_thread(
            manager.remember, "User prefers Neovim", MemoryType.PREFERENCE, 0.8
        )
        assert mem.content == "User prefers Neovim"
        retrieved = await asyncio.to_thread(manager.get_memory, mem.id)
        assert retrieved is not None
        assert retrieved.content == "User prefers Neovim"

    async def test_search_via_to_thread(self, manager):
        await asyncio.to_thread(manager.remember, "Python is the language", MemoryType.FACT, 0.7)
        results = await asyncio.to_thread(manager.search, "Python language", 5)
        assert len(results) >= 1
        assert "Python" in results[0].content

    async def test_conversation_via_to_thread(self, manager):
        conv = manager.conversation
        await asyncio.to_thread(conv.add_user_message, "Hello Jarvis", "evt1")
        await asyncio.to_thread(conv.add_assistant_message, "Hello!", "evt1_resp")
        messages = await asyncio.to_thread(conv.recent_messages, 10)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    async def test_forget_via_to_thread(self, manager):
        mem = await asyncio.to_thread(manager.remember, "temporary")
        await asyncio.to_thread(manager.forget, mem.id)
        result = await asyncio.to_thread(manager.get_memory, mem.id)
        assert result is None

    async def test_clear_session_via_to_thread(self, manager):
        conv = manager.conversation
        await asyncio.to_thread(conv.add_user_message, "msg1")
        await asyncio.to_thread(manager.clear_session, conv.session_id)
        messages = await asyncio.to_thread(conv.recent_messages, 10)
        assert len(messages) == 0


class TestConcurrentWorkerThreads:
    """Multiple worker threads accessing the database concurrently via gather."""

    async def test_concurrent_remembers(self, manager):
        """Multiple remember() calls from different threads simultaneously."""
        tasks = [
            asyncio.to_thread(manager.remember, f"memory {i}", MemoryType.NOTE, 0.5)
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 20
        assert all(r.content.startswith("memory ") for r in results)

        # All 20 should be in the database
        all_mems = await asyncio.to_thread(manager.store.all_memories, 100)
        assert len(all_mems) == 20

    async def test_concurrent_reads_and_writes(self, manager):
        """Interleaved reads and writes from different threads."""
        # Seed some data
        for i in range(5):
            await asyncio.to_thread(manager.remember, f"seed {i}", MemoryType.FACT, 0.5)

        async def writer(idx):
            return await asyncio.to_thread(manager.remember, f"concurrent {idx}", MemoryType.NOTE, 0.5)

        async def reader():
            return await asyncio.to_thread(manager.search, "seed", 5)

        # Run writers and readers concurrently
        tasks = [writer(i) for i in range(10)] + [reader() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # Writers return Memory, readers return list
        writer_results = results[:10]
        reader_results = results[10:]
        assert all(r is not None and r.content.startswith("concurrent ") for r in writer_results)
        assert all(isinstance(r, list) for r in reader_results)

    async def test_concurrent_conversation_messages(self, manager):
        """Multiple conversation writes from different threads."""
        conv = manager.conversation

        async def add_msg(idx):
            await asyncio.to_thread(conv.add_user_message, f"msg {idx}", f"evt_{idx}")

        tasks = [add_msg(i) for i in range(10)]
        await asyncio.gather(*tasks)

        messages = await asyncio.to_thread(conv.recent_messages, 50)
        assert len(messages) == 10

    async def test_concurrent_search_during_writes(self, manager):
        """Search doesn't fail while writes are happening."""
        await asyncio.to_thread(manager.remember, "existing memory about Python", MemoryType.FACT)

        async def do_write(idx):
            await asyncio.to_thread(manager.remember, f"new item {idx}", MemoryType.NOTE)

        async def do_search():
            return await asyncio.to_thread(manager.search, "Python", 5)

        tasks = [do_write(i) for i in range(5)] + [do_search() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        search_results = results[5:]
        for r in search_results:
            assert isinstance(r, list)
            assert any("Python" in m.content for m in r)

    async def test_no_sqlite_programming_error(self, tmp_path):
        """The old single-connection pattern would raise ProgrammingError.
        Thread-local connections avoid this."""
        store = SQLiteStore(tmp_path / "thread_test.db")
        manager = MemoryManager(store)
        manager.start_session()

        errors = []

        async def worker(idx):
            try:
                await asyncio.to_thread(manager.remember, f"item {idx}", MemoryType.NOTE, 0.5)
                await asyncio.to_thread(manager.search, "item", 3)
            except Exception as e:
                errors.append(e)

        await asyncio.gather(*[worker(i) for i in range(20)])
        assert len(errors) == 0, f"Errors encountered: {errors}"
        manager.close()


class TestConcurrentIdempotency:
    """Verify idempotency holds under concurrent writes."""

    async def test_concurrent_duplicate_source_event_id(self, manager):
        """Same source_event_id from multiple threads → exactly one message."""
        conv = manager.conversation

        async def add_duplicate():
            await asyncio.to_thread(conv.add_user_message, "duplicate msg", "same_event")

        # Fire 10 concurrent attempts with the same source_event_id
        await asyncio.gather(*[add_duplicate() for _ in range(10)])

        messages = await asyncio.to_thread(conv.recent_messages, 50)
        assert len(messages) == 1
        assert messages[0].content == "duplicate msg"
        assert messages[0].source_event_id == "same_event"

    async def test_user_and_assistant_same_event_both_persist(self, manager):
        """User and assistant messages for same event have different source IDs."""
        conv = manager.conversation
        await asyncio.to_thread(conv.add_user_message, "question", "evt_abc")
        await asyncio.to_thread(conv.add_assistant_message, "answer", "evt_abc_response")

        messages = await asyncio.to_thread(conv.recent_messages, 10)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"


class TestAsyncNonBlocking:
    """Verify database operations don't block the event loop."""

    async def test_db_operations_dont_block_loop(self, manager):
        loop_alive = False

        async def background():
            nonlocal loop_alive
            await asyncio.sleep(0.01)
            loop_alive = True

        bg = asyncio.create_task(background())

        # DB operation in thread
        await asyncio.to_thread(manager.remember, "test content", MemoryType.NOTE, 0.5)
        await asyncio.to_thread(manager.search, "test", 5)

        await bg
        assert loop_alive is True


class TestConnectionLifecycle:
    """Verify connection registry and close() behavior."""

    async def test_close_prevents_further_use(self, tmp_path):
        """After close(), any operation raises RuntimeError."""
        store = SQLiteStore(tmp_path / "lifecycle.db")
        manager = MemoryManager(store)
        manager.start_session()
        manager.remember("before close")
        manager.close()

        with pytest.raises(RuntimeError, match="closed"):
            store.get("anything")

    async def test_close_closes_worker_thread_connections(self, tmp_path):
        """close() closes connections created by worker threads."""
        store = SQLiteStore(tmp_path / "lifecycle2.db")
        manager = MemoryManager(store)
        manager.start_session()

        # Create connections on worker threads
        await asyncio.to_thread(manager.remember, "from worker 1", MemoryType.NOTE, 0.5)
        await asyncio.to_thread(manager.remember, "from worker 2", MemoryType.NOTE, 0.5)

        # Registry should have connections
        assert len(store._connections) >= 1

        manager.close()

        # All connections closed
        assert len(store._connections) == 0
        assert store._closed is True

    async def test_operations_after_close_fail_from_worker(self, tmp_path):
        """Worker threads cannot use the store after close."""
        store = SQLiteStore(tmp_path / "lifecycle3.db")
        manager = MemoryManager(store)
        manager.start_session()
        await asyncio.to_thread(manager.remember, "pre-close")
        manager.close()

        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.to_thread(manager.remember, "post-close")


class TestInMemoryBehavior:
    """Verify :memory: works correctly with shared-cache URI."""

    def test_memory_db_single_thread_works(self):
        """Basic :memory: usage on one thread works."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()
        mem = manager.remember("in memory")
        assert manager.get_memory(mem.id) is not None
        manager.close()

    async def test_memory_db_cross_thread_visibility(self):
        """Data written on one thread is visible from another (shared-cache)."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()

        # Write on main thread
        mem = manager.remember("visible to workers")

        # Read from a worker thread
        result = await asyncio.to_thread(manager.get_memory, mem.id)
        assert result is not None
        assert result.content == "visible to workers"
        manager.close()

    async def test_memory_db_worker_write_visible_to_main(self):
        """Data written by a worker thread is visible on the main thread."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()

        # Write from worker
        mem = await asyncio.to_thread(manager.remember, "from worker", MemoryType.NOTE, 0.5)

        # Read from main thread
        result = manager.get_memory(mem.id)
        assert result is not None
        assert result.content == "from worker"
        manager.close()

    async def test_memory_db_concurrent_operations(self):
        """Concurrent operations on :memory: work with sequential writes.

        Note: shared-cache :memory: uses table-level locking (not WAL), so
        truly concurrent writes can conflict. This is acceptable because
        the real application uses file-backed WAL databases for concurrency.
        :memory: is intended for tests with limited concurrency.
        """
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()

        # Sequential writes are fine
        for i in range(10):
            await asyncio.to_thread(manager.remember, f"mem {i}", MemoryType.NOTE, 0.5)

        all_mems = manager.store.all_memories(100)
        assert len(all_mems) == 10
        manager.close()

    def test_file_backed_uses_wal(self, tmp_path):
        """File-backed databases get WAL mode."""
        store = SQLiteStore(tmp_path / "wal_test.db")
        conn = store._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        store.close()

    def test_memory_db_no_wal(self):
        """:memory: databases don't use WAL (not applicable)."""
        store = SQLiteStore(":memory:")
        conn = store._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "memory"
        store.close()
