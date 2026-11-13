"""FTS5 retrieval, migration, explicit memories and write-path guarantees."""
import concurrent.futures
import sqlite3
import time

import pytest

from app.memory.manager import MemoryManager
from app.memory.sqlite_store import SQLiteStore, fts_query
from app.memory.types import ConversationMessage, ConversationSession, Memory, MemoryType

# Schema as shipped before the FTS migration (commit 9780e4f).
_OLD_SCHEMA = """
CREATE TABLE conversation_sessions (id TEXT PRIMARY KEY, started_at REAL NOT NULL, last_activity_at REAL NOT NULL);
CREATE TABLE conversation_messages (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL, timestamp REAL NOT NULL, source_event_id TEXT,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(id));
CREATE UNIQUE INDEX idx_messages_source_event_unique
    ON conversation_messages(source_event_id) WHERE source_event_id IS NOT NULL;
CREATE TABLE memories (
    id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5, source_event_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}');
CREATE INDEX idx_messages_session_id ON conversation_messages(session_id);
CREATE INDEX idx_messages_timestamp ON conversation_messages(timestamp);
CREATE INDEX idx_memories_type ON memories(type);
CREATE INDEX idx_memories_created ON memories(created_at);
"""


def _old_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    now = time.time()
    conn.execute("INSERT INTO conversation_sessions VALUES ('s_old', ?, ?)", (now - 86400, now - 86400))
    conn.execute(
        "INSERT INTO conversation_messages VALUES ('m1', 's_old', 'user', 'my favourite colour is teal', ?, 'evt_old')",
        (now - 86400,),
    )
    conn.execute(
        "INSERT INTO memories VALUES ('mem1', 'fact', 'the boiler code is 4471', ?, ?, 0.5, NULL, '{}')",
        (now - 86400, now - 86400),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture
def manager(store):
    m = MemoryManager(store)
    m.start_session()
    return m


def _past_session(store, texts, *, age_days=1.0):
    now = time.time() - age_days * 86400
    sess = ConversationSession(id=f"past_{age_days}_{len(texts)}", started_at=now, last_activity_at=now)
    store.create_session(sess)
    for i, (role, text) in enumerate(texts):
        store.add_message(ConversationMessage(
            id=f"{sess.id}_{i}", session_id=sess.id, role=role, content=text, timestamp=now + i,
        ))
    return sess


class TestMigration:

    def test_old_schema_migrates_in_place_and_is_idempotent(self, tmp_path):
        db = tmp_path / "old.db"
        _old_db(db)

        for _ in range(2):  # second open must be a no-op
            s = SQLiteStore(db)
            assert s.get("mem1").content == "the boiler code is 4471"
            assert [m.content for m in s.get_messages("s_old")] == ["my favourite colour is teal"]
            assert [m.id for m in s.search("boiler code")] == ["mem1"]
            assert s.search_message_rows("favourite colour", 5, exclude_session_id=None)
            s.close()

        conn = sqlite3.connect(db)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0] == 1
        assert conn.execute("SELECT norm_content FROM memories").fetchone()[0] == "the boiler code is 4471"
        idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_messages_session_ts" in idx and "idx_messages_session_id" not in idx
        conn.close()

    def test_wal_and_synchronous_normal(self, store):
        conn = store._get_conn()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL


class TestFtsSync:

    def test_insert_update_delete(self, store):
        mem = Memory.create(content="the garage door code is 1234", memory_type=MemoryType.FACT)
        store.add(mem)
        assert [m.id for m in store.search("garage")] == [mem.id]

        conn = store._get_conn()
        conn.execute("UPDATE memories SET content = 'the shed key is under the mat' WHERE id = ?", (mem.id,))
        conn.commit()
        assert store.search("garage") == []
        assert [m.id for m in store.search("shed")] == [mem.id]

        store.delete(mem.id)
        assert store.search("shed") == []

    def test_message_delete_via_clear_session(self, store):
        sess = _past_session(store, [("user", "the plumber arrives thursday")])
        assert store.search_message_rows("plumber", 5, exclude_session_id=None)
        store.clear_session(sess.id)
        assert store.search_message_rows("plumber", 5, exclude_session_id=None) == []


class TestHostileQueries:

    @pytest.mark.parametrize("q", [
        "", "   ", '"', '""', "'", "OR", "AND NOT", "NEAR(a b)", "*", "^start", "col:value",
        "a - b", "(unbalanced", "the is a of", '"quoted" OR "stuff"', "x" * 5000, "émoji 🎉 ünïcode",
    ])
    def test_never_raises(self, manager, q):
        manager.remember("some stored fact about quoted stuff")
        assert isinstance(manager.retrieve_context(q, limit=5), list)
        assert isinstance(manager.search(q), list)

    def test_stopwords_only_gives_no_query(self):
        assert fts_query("what is the") is None
        assert fts_query('say "hi" OR not') is None
        assert fts_query("Wifi PASSWORD wifi") == '"wifi" OR "password"'


class TestRetrieveContext:

    def test_cross_session_excludes_current_session(self, manager, store):
        _past_session(store, [("user", "my sister lives in Lisbon"), ("assistant", "Lisbon is lovely")])
        manager.conversation.add_user_message("does my sister like Lisbon", "evt_now")

        lines = manager.retrieve_context("where does my sister live", limit=5)
        assert any("my sister lives in Lisbon" in line for line in lines)
        assert not any("does my sister like Lisbon" in line for line in lines)

    def test_cross_session_off(self, manager, store):
        _past_session(store, [("user", "my sister lives in Lisbon")])
        manager.cross_session = False
        assert manager.retrieve_context("sister Lisbon", limit=5) == []

    def test_provenance_tags_and_dates(self, manager, store):
        _past_session(store, [("user", "the dentist appointment is on friday")], age_days=3)
        manager.remember("dentist is Dr Rao on Elm street", memory_type=MemoryType.FACT)
        lines = manager.retrieve_context("dentist", limit=5)
        day = time.strftime("%Y-%m-%d", time.localtime(time.time() - 3 * 86400))
        today = time.strftime("%Y-%m-%d")
        assert f"[memory, {today}] dentist is Dr Rao on Elm street" in lines
        assert f"[earlier conversation, {day}, user] the dentist appointment is on friday" in lines

    def test_word_match_beats_substring_coincidence(self, manager):
        manager.remember("the party was smart and fun")
        manager.remember("I love modern art museums")
        lines = manager.retrieve_context("art", limit=5)
        assert len(lines) == 1 and "art museums" in lines[0]

    def test_facts_differing_in_a_number_are_kept(self, manager):
        manager.remember("the gate code is 4321")
        manager.remember("the gate code is 4322")
        assert len(manager.retrieve_context("gate code", limit=5)) == 2

    def test_near_duplicates_removed(self, manager, store):
        _past_session(store, [("user", "My wifi password is hunter2."), ("user", "my wifi password is hunter2")])
        manager.remember("my wifi password is hunter2!")
        lines = manager.retrieve_context("wifi password", limit=5)
        assert len(lines) == 1

    def test_recency_breaks_ties(self, manager, store):
        _past_session(store, [("user", "the meeting room is B12")], age_days=60)
        _past_session(store, [("user", "the meeting room is C7")], age_days=1)
        lines = manager.retrieve_context("meeting room", limit=5)
        assert "C7" in lines[0]

    def test_limit_respected(self, manager):
        for tool in ("rake", "hoe", "spade", "shears", "trowel", "hose"):
            manager.remember(f"the {tool} is in the gardening shed")
        assert len(manager.retrieve_context("gardening", limit=3)) == 3
        assert manager.retrieve_context("gardening", limit=0) == []

    def test_concurrent_retrieve_and_add(self, manager, store):
        _past_session(store, [("user", f"older note {i} about the camping trip") for i in range(50)])
        conv = manager.conversation

        def write(i):
            return conv.add_user_message(f"camping gear item {i}", f"evt_{i}")

        def read(_):
            return manager.retrieve_context("camping trip gear", limit=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(write, i) for i in range(40)] + [pool.submit(read, i) for i in range(40)]
            results = [f.result() for f in futures]
        assert all(results[40:])
        assert len(conv.recent_messages(100)) == 40
        # Current-session writes never leak into retrieval.
        assert not any("gear item" in line for r in results[40:] for line in r)

    def test_latency_10k_rows(self, manager, store):
        words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
                 "mike november oscar papa quebec romeo sierra tango uniform victor").split()
        sess = _past_session(store, [])
        conn = store._get_conn()
        now = time.time()
        conn.executemany(
            "INSERT INTO conversation_messages (id, session_id, role, content, timestamp) VALUES (?, ?, 'user', ?, ?)",
            [(f"r{i}", sess.id, " ".join(words[(i * k) % len(words)] for k in (1, 3, 7, 11, 13)) + f" item{i}",
              now - i) for i in range(10_000)],
        )
        conn.commit()
        manager.retrieve_context("warm up", limit=5)
        times = []
        for q in ("alpha mike", "tango sierra item42", "what about golf", "romeo and juliet"):
            t = time.perf_counter()
            assert manager.retrieve_context(q, limit=5)
            times.append(time.perf_counter() - t)
        assert max(times) < 0.2  # generous for loaded CI; benchmark reports the real number


class TestMaybeRemember:

    @pytest.mark.parametrize("text,content,mtype", [
        ("Jarvis, remember that my car is parked on level 3.", "my car is parked on level 3", MemoryType.FACT),
        ("remember my wifi password is hunter2", "my wifi password is hunter2", MemoryType.FACT),
        ("Hey Jarvis please remember the gate code is 4321", "the gate code is 4321", MemoryType.FACT),
        ("Don’t forget that the bins go out on Tuesday", "the bins go out on Tuesday", MemoryType.FACT),
        ("don't forget to call mom tomorrow", "Reminder: call mom tomorrow", MemoryType.TASK),
        ("Note that the meeting moved to 4pm", "the meeting moved to 4pm", MemoryType.FACT),
        ("can you remember that I prefer tea over coffee", "I prefer tea over coffee", MemoryType.FACT),
    ])
    def test_positive(self, manager, text, content, mtype):
        mem = manager.maybe_remember(text, "evt_1")
        assert mem is not None
        assert (mem.content, mem.type) == (content, mtype)
        assert mem.source_event_id == "evt_1"
        assert mem.metadata["session_id"] == manager.conversation.session_id
        assert manager.get_memory(mem.id) is not None

    @pytest.mark.parametrize("text", [
        "do you remember what my wifi password is?",
        "remember when we went to Paris",
        "remember that time we went to Paris",
        "I don't remember that",
        "please remember that",
        "remember that?",
        "what is the weather today",
        "I can't remember my password",
        "remember to",
        "",
    ])
    def test_negative(self, manager, text):
        assert manager.maybe_remember(text, "evt_x") is None
        assert manager.store.all_memories() == []

    def test_dedup_returns_existing(self, manager):
        a = manager.maybe_remember("remember that my car is parked on level 3", "e1")
        b = manager.maybe_remember("Jarvis, remember that my car is parked on level 3!", "e2")
        assert b.id == a.id
        assert len(manager.store.all_memories()) == 1

    def test_concurrent_dedup(self, manager):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = set(pool.map(
                lambda i: manager.maybe_remember("remember that the spare key is in the blue pot", f"e{i}").id,
                range(16),
            ))
        assert len(ids) == 1
        assert len(manager.store.all_memories()) == 1

    def test_remembered_fact_is_retrieved(self, manager):
        manager.maybe_remember("remember that the spare key is in the blue pot", "e1")
        assert manager.retrieve_context("where is the spare key", limit=3)[0].endswith(
            "the spare key is in the blue pot"
        )


class TestWritePath:

    def test_add_user_message_idempotent_returns_existing(self, manager):
        conv = manager.conversation
        first = conv.add_user_message("hello there", "evt_dup")
        second = conv.add_user_message("hello there", "evt_dup")
        assert second.id == first.id
        assert len(conv.recent_messages(10)) == 1

    def test_one_commit_per_message(self, manager, store):
        conn = store._get_conn()
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        try:
            manager.conversation.add_user_message("one transaction please", "evt_tx")
            manager.conversation.add_user_message("one transaction please", "evt_tx")
        finally:
            conn.set_trace_callback(None)
        assert sum(s.strip().upper() == "COMMIT" for s in statements) == 2
        assert sum(s.strip().upper().startswith("BEGIN") for s in statements) <= 2

    def test_session_activity_updated_in_same_write(self, manager, store):
        msg = manager.conversation.add_user_message("tick", "evt_tick")
        assert store.get_session(manager.conversation.session_id).last_activity_at == msg.timestamp
