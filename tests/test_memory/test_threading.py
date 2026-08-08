"""Tests for SQLite thread safety and async concurrency.

Verifies that database operations work correctly when called through
asyncio.to_thread() from multiple worker threads simultaneously.
Tests connection lifecycle, cross-thread shutdown, and in-memory behavior.
"""
import asyncio
import concurrent.futures
import threading
import time

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

        all_mems = await asyncio.to_thread(manager.store.all_memories, 100)
        assert len(all_mems) == 20

    async def test_concurrent_reads_and_writes(self, manager):
        """Interleaved reads and writes from different threads."""
        for i in range(5):
            await asyncio.to_thread(manager.remember, f"seed {i}", MemoryType.FACT, 0.5)

        async def writer(idx):
            return await asyncio.to_thread(manager.remember, f"concurrent {idx}", MemoryType.NOTE, 0.5)

        async def reader():
            return await asyncio.to_thread(manager.search, "seed", 5)

        tasks = [writer(i) for i in range(10)] + [reader() for _ in range(5)]
        results = await asyncio.gather(*tasks)

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
        """close() closes connections created by worker threads (cross-thread)."""
        store = SQLiteStore(tmp_path / "lifecycle2.db")
        manager = MemoryManager(store)
        manager.start_session()

        await asyncio.to_thread(manager.remember, "from worker 1", MemoryType.NOTE, 0.5)
        await asyncio.to_thread(manager.remember, "from worker 2", MemoryType.NOTE, 0.5)

        assert len(store._connections) >= 1

        manager.close()

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

    async def test_cross_thread_close_no_thread_affinity_error(self, tmp_path):
        """close() from main thread does not raise ProgrammingError for
        connections created by worker threads."""
        store = SQLiteStore(tmp_path / "cross_close.db")
        manager = MemoryManager(store)
        manager.start_session()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futs = [
                pool.submit(manager.remember, f"item {i}", MemoryType.NOTE, 0.5)
                for i in range(3)
            ]
            concurrent.futures.wait(futs)
            for f in futs:
                f.result()

        num_conns = len(store._connections)
        assert num_conns >= 2

        manager.close()
        assert len(store._connections) == 0

    async def test_connections_actually_closed(self, tmp_path):
        """Registered connections are genuinely closed after shutdown."""
        store = SQLiteStore(tmp_path / "actual_close.db")
        manager = MemoryManager(store)
        manager.start_session()

        conn = store._get_conn()
        assert conn in store._connections

        manager.close()

        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    async def test_multiple_worker_connections_all_closed(self, tmp_path):
        """close() handles connections from many worker threads."""
        store = SQLiteStore(tmp_path / "multi_worker.db")
        manager = MemoryManager(store)
        manager.start_session()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futs = [
                pool.submit(manager.remember, f"data {i}", MemoryType.NOTE, 0.5)
                for i in range(5)
            ]
            concurrent.futures.wait(futs)
            for f in futs:
                f.result()

        pre_close_count = len(store._connections)
        assert pre_close_count >= 2

        conns_before = list(store._connections)

        manager.close()
        assert len(store._connections) == 0

        for c in conns_before:
            with pytest.raises(Exception):
                c.execute("SELECT 1")

    def test_close_idempotent(self, tmp_path):
        """close() can be called multiple times safely."""
        store = SQLiteStore(tmp_path / "idempotent.db")
        manager = MemoryManager(store)
        manager.start_session()
        manager.remember("data")

        manager.close()
        manager.close()
        manager.close()

        assert store._closed is True
        assert len(store._connections) == 0

    def test_close_from_worker_thread(self, tmp_path):
        """close() works correctly when called from a worker thread."""
        store = SQLiteStore(tmp_path / "worker_close.db")
        manager = MemoryManager(store)
        manager.start_session()
        manager.remember("data")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            f = pool.submit(manager.close)
            f.result()

        assert store._closed is True
        assert len(store._connections) == 0

        with pytest.raises(RuntimeError, match="closed"):
            store.get("anything")


class TestLifecycleRace:
    """Deterministic race tests using threading.Event for forced ordering.

    These tests guarantee specific interleavings that ordinary concurrent
    tests may never reproduce.
    """

    def test_close_waits_for_active_op(self, tmp_path):
        """close() blocks until an in-flight operation finishes.

        Forced ordering:
          worker enters operation → worker pauses → main calls close() →
          close enters CLOSING → worker finishes → close completes.
        """
        store = SQLiteStore(tmp_path / "race1.db")
        manager = MemoryManager(store)
        manager.start_session()

        op_started = threading.Event()
        op_may_finish = threading.Event()
        op_result = [None]

        def slow_operation():
            store._begin_op()
            op_started.set()
            try:
                op_may_finish.wait()
                conn = store._get_conn()
                conn.execute(
                    "INSERT OR IGNORE INTO memories (id, type, content, created_at, updated_at, importance, source_event_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("test_id", "note", "race data", time.time(), time.time(), 0.5, None, "{}"),
                )
                conn.commit()
                op_result[0] = "success"
            finally:
                store._end_op()

        worker = threading.Thread(target=slow_operation)
        worker.start()

        # Wait for worker to be inside the operation
        op_started.wait()

        # close() should now block because active_ops > 0
        close_done = threading.Event()

        def do_close():
            store.close()
            close_done.set()

        closer = threading.Thread(target=do_close)
        closer.start()

        # Give close() a moment to enter CLOSING state
        time.sleep(0.05)
        assert store._closing is True
        assert not close_done.is_set()  # close is still waiting

        # Let the worker finish
        op_may_finish.set()
        worker.join(timeout=5)

        # close should now complete
        closer.join(timeout=5)
        assert close_done.is_set()
        assert store._closed is True
        assert op_result[0] == "success"

    def test_new_op_rejected_during_close(self, tmp_path):
        """New operations are rejected once CLOSING state is entered.

        Forced ordering:
          worker A enters op → main calls close() → enters CLOSING →
          worker B attempts op → worker B rejected → worker A finishes →
          close completes.
        """
        store = SQLiteStore(tmp_path / "race2.db")
        manager = MemoryManager(store)
        manager.start_session()

        op_a_started = threading.Event()
        op_a_may_finish = threading.Event()
        close_started = threading.Event()
        op_b_result = [None]

        def operation_a():
            store._begin_op()
            op_a_started.set()
            try:
                op_a_may_finish.wait()
            finally:
                store._end_op()

        def operation_b():
            close_started.wait()
            time.sleep(0.02)  # ensure close has entered CLOSING
            try:
                store._begin_op()
                store._end_op()
                op_b_result[0] = "admitted"
            except RuntimeError as e:
                op_b_result[0] = f"rejected: {e}"

        thread_a = threading.Thread(target=operation_a)
        thread_b = threading.Thread(target=operation_b)
        thread_a.start()

        op_a_started.wait()

        def do_close():
            close_started.set()
            store.close()

        closer = threading.Thread(target=do_close)
        closer.start()

        thread_b.start()
        thread_b.join(timeout=5)

        # B must be rejected
        assert op_b_result[0] is not None
        assert "rejected" in op_b_result[0]

        # Let A finish so close() can complete
        op_a_may_finish.set()
        thread_a.join(timeout=5)
        closer.join(timeout=5)

        assert store._closed is True

    def test_op_completes_before_close_instant(self, tmp_path):
        """If no operations are active, close() returns immediately."""
        store = SQLiteStore(tmp_path / "race3.db")
        manager = MemoryManager(store)
        manager.start_session()
        manager.remember("data")

        start = time.time()
        manager.close()
        elapsed = time.time() - start

        assert store._closed is True
        assert elapsed < 1.0  # essentially instant

    def test_close_does_not_close_active_connection(self, tmp_path):
        """A connection being used by an active operation is not closed
        until the operation finishes."""
        store = SQLiteStore(tmp_path / "race4.db")
        manager = MemoryManager(store)
        manager.start_session()

        op_started = threading.Event()
        op_may_finish = threading.Event()
        conn_worked = [False]

        def operation_with_conn():
            store._begin_op()
            op_started.set()
            try:
                op_may_finish.wait()
                conn = store._get_conn()
                # This must work — connection must not be closed yet
                conn.execute("SELECT 1").fetchone()
                conn_worked[0] = True
            finally:
                store._end_op()

        worker = threading.Thread(target=operation_with_conn)
        worker.start()
        op_started.wait()

        close_done = threading.Event()
        closer = threading.Thread(target=lambda: (store.close(), close_done.set()))
        closer.start()

        time.sleep(0.05)
        # close is waiting; connection must still work for the active op
        op_may_finish.set()
        worker.join(timeout=5)
        closer.join(timeout=5)

        assert conn_worked[0] is True
        assert store._closed is True


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

        mem = manager.remember("visible to workers")

        result = await asyncio.to_thread(manager.get_memory, mem.id)
        assert result is not None
        assert result.content == "visible to workers"
        manager.close()

    async def test_memory_db_worker_write_visible_to_main(self):
        """Data written by a worker thread is visible on the main thread."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()

        mem = await asyncio.to_thread(manager.remember, "from worker", MemoryType.NOTE, 0.5)

        result = manager.get_memory(mem.id)
        assert result is not None
        assert result.content == "from worker"
        manager.close()

    async def test_memory_db_sequential_cross_thread(self):
        """Sequential cross-thread operations on :memory: work.

        Shared-cache :memory: uses table-level locking (not WAL), so the real
        application uses file-backed WAL for concurrent writes. :memory: is
        for tests with limited concurrency.
        """
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()

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

    def test_memory_db_keeper_closed_on_shutdown(self):
        """The keeper connection is closed when the store shuts down."""
        store = SQLiteStore(":memory:")
        assert store._keeper is not None
        keeper_ref = store._keeper
        store.close()
        assert store._keeper is None
        # Keeper should be genuinely closed
        with pytest.raises(Exception):
            keeper_ref.execute("SELECT 1")

    async def test_memory_db_cross_thread_close(self):
        """Cross-thread close works for :memory: stores too."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()

        await asyncio.to_thread(manager.remember, "data", MemoryType.NOTE, 0.5)
        manager.close()

        assert store._closed is True
        assert store._keeper is None
        assert len(store._connections) == 0
