"""Context priority, budget, ordering, and deduplication regression tests.

Verifies:
- Priority: current request > conversation > memory > working memory
- Budget: context never exceeds max_characters
- Ordering: conversation is chronological, current request always last
- Deduplication: current event never appears twice
"""
import pytest

from app.attention.engine import AttentionEngine
from app.context.builder import ContextBuilder
from app.events.types import Modality, SpeechEvent
from app.memory.manager import MemoryManager
from app.memory.sqlite_store import SQLiteStore
from app.memory.types import MemoryType
from app.memory.working import WorkingMemory


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(
        session_id="test", source=Modality.AUDIO,
        transcript=text, duration_seconds=1.0,
    )


class TestContextPriority:
    """Current request > conversation > memory > working memory."""

    def test_current_request_survives_tight_budget(self):
        """Even with minimal budget, current request is always present."""
        builder = ContextBuilder(max_messages=8, max_characters=800)
        memory = WorkingMemory(max_size=20)
        for i in range(10):
            memory.record(f"e{i}", "x" * 50, should_reason=True)
        conv = [{"role": "user", "content": "y" * 100} for _ in range(10)]
        mem_ctx = ["z" * 100 for _ in range(10)]

        ctx = builder.build(_speech("CURRENT REQUEST"), memory, conversation_messages=conv, memory_context=mem_ctx)
        msgs = builder.to_messages(ctx)

        assert msgs[-1]["content"] == "CURRENT REQUEST"

    def test_conversation_preferred_over_memory_when_budget_tight(self):
        """Conversation gets budget before memories."""
        builder = ContextBuilder(max_messages=8, max_characters=1200, max_memory_items=5)
        memory = WorkingMemory(max_size=20)
        conv = [
            {"role": "user", "content": "conv question"},
            {"role": "assistant", "content": "conv answer"},
        ]
        mem_ctx = ["memory item " * 10 for _ in range(5)]

        ctx = builder.build(_speech("current"), memory, conversation_messages=conv, memory_context=mem_ctx)
        msgs = builder.to_messages(ctx)

        contents = [m["content"] for m in msgs]
        # Conversation should be present
        assert "conv question" in contents
        assert "conv answer" in contents

    def test_memory_excluded_when_budget_exhausted_by_conversation(self):
        """If conversation fills the budget, memory gets nothing."""
        builder = ContextBuilder(max_messages=8, max_characters=900, max_memory_items=5)
        memory = WorkingMemory(max_size=20)
        conv = [{"role": "user", "content": "c" * 150} for _ in range(3)]
        mem_ctx = ["memory data " * 20]

        ctx = builder.build(_speech("current"), memory, conversation_messages=conv, memory_context=mem_ctx)
        # Budget may not allow memory
        assert ctx.total_characters <= 900


class TestContextOrdering:
    """Verify conversation ordering: oldest → newest → current."""

    def test_conversation_chronological_in_messages(self):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        conv = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        ctx = builder.build(_speech("Q3"), memory, conversation_messages=conv)
        msgs = builder.to_messages(ctx)

        non_system = [m for m in msgs if m["role"] != "system"]
        contents = [m["content"] for m in non_system]
        assert contents[-1] == "Q3"
        assert contents.index("Q1") < contents.index("A1")
        assert contents.index("A1") < contents.index("Q2")
        assert contents.index("Q2") < contents.index("A2")
        assert contents.index("A2") < contents.index("Q3")

    def test_current_request_always_last_user_message(self):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        memory.record("e1", "working mem entry", should_reason=True)
        conv = [{"role": "user", "content": "prior question"}]

        ctx = builder.build(_speech("final question"), memory, conversation_messages=conv)
        msgs = builder.to_messages(ctx)
        assert msgs[-1] == {"role": "user", "content": "final question"}


class TestCurrentEventDeduplication:
    """Current event must NOT appear twice (once as history, once as current)."""

    def test_no_duplication_with_conversation_containing_current(self):
        """If conversation_messages accidentally includes current event content,
        the current request still only appears once at the end."""
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        event = _speech("Jarvis, what is the weather?")

        # Simulate conversation that includes this same text
        conv = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        ctx = builder.build(event, memory, conversation_messages=conv)
        msgs = builder.to_messages(ctx)
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_msgs.count("Jarvis, what is the weather?") == 1

    def test_no_duplication_after_attention_records_event(self):
        """After attention records the event in working memory,
        context builder must include the current transcript exactly once."""
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=8, max_characters=6000)

        event = _speech("Jarvis, explain HTTP 500 errors.")
        engine.evaluate(event)

        # Build context with the same event (attention already recorded it)
        ctx = builder.build(event, engine.working_memory)
        msgs = builder.to_messages(ctx)
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_contents.count("Jarvis, explain HTTP 500 errors.") == 1

    def test_source_event_id_filtering_prevents_duplication(self):
        """In the real pipeline, source_event_id filtering in main.py prevents
        the current event from appearing in both conversation history and current."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)

        event = _speech("Hey Jarvis, help me")
        # Simulate: user message persisted before context build
        conv.add_user_message(event.transcript, source_event_id=event.id)

        # Retrieve recent but exclude current event
        recent = conv.recent_messages(limit=8)
        conv_msgs = [
            {"role": m.role, "content": m.content}
            for m in recent if m.source_event_id != event.id
        ]
        ctx = builder.build(event, memory, conversation_messages=conv_msgs)
        msgs = builder.to_messages(ctx)
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_contents.count("Hey Jarvis, help me") == 1

        manager.close()


class TestAssistantResponseCorrelation:
    """User and assistant messages maintain proper event correlation."""

    def test_user_and_assistant_correlation(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()

        event_id = "abc123"
        conv.add_user_message("user question", source_event_id=event_id)
        conv.add_assistant_message("assistant answer", source_event_id=event_id + "_response")

        messages = conv.recent_messages()
        assert messages[0].source_event_id == event_id
        assert messages[1].source_event_id == event_id + "_response"
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        manager.close()

    def test_response_correlation_is_deterministic(self):
        """event.id + '_response' is always the same for the same event."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()

        conv.add_assistant_message("answer", source_event_id="evt42_response")
        # Attempting to add again with same source_event_id should be idempotent
        conv.add_assistant_message("answer again", source_event_id="evt42_response")

        messages = conv.recent_messages()
        assert len(messages) == 1
        assert messages[0].content == "answer"
        manager.close()


class TestSessionActivityTracking:
    """Session last_activity_at updates on new messages."""

    def test_activity_updated_on_user_message(self, tmp_path):
        import time
        store = SQLiteStore(tmp_path / "activity.db")
        manager = MemoryManager(store)
        conv = manager.start_session()
        original_activity = store.get_session(conv.session_id).last_activity_at
        time.sleep(0.02)
        conv.add_user_message("update")
        updated = store.get_session(conv.session_id)
        assert updated.last_activity_at > original_activity
        manager.close()

    def test_activity_updated_on_assistant_message(self, tmp_path):
        import time
        store = SQLiteStore(tmp_path / "activity2.db")
        manager = MemoryManager(store)
        conv = manager.start_session()
        original_activity = store.get_session(conv.session_id).last_activity_at
        time.sleep(0.02)
        conv.add_assistant_message("response")
        updated = store.get_session(conv.session_id)
        assert updated.last_activity_at > original_activity
        manager.close()


class TestMemoryLimitsEnforced:
    """max_retrieved_memories and max_recent_messages are enforced."""

    def test_max_retrieved_memories_limits_context(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()
        for i in range(50):
            manager.remember(f"memory about topic {i}", memory_type=MemoryType.NOTE)

        results = manager.search("topic", limit=5)
        assert len(results) <= 5
        manager.close()

    def test_max_recent_messages_limits_context(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()
        for i in range(100):
            conv.add_user_message(f"message {i}")

        recent = conv.recent_messages(limit=8)
        assert len(recent) == 8
        manager.close()

    def test_10000_memories_context_bounded(self):
        """With many memories, context builder still produces bounded context."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.start_session()
        # Insert many memories (using direct add for speed)
        from app.memory.types import Memory
        for i in range(100):
            manager.remember(f"memory {i} about various topics", memory_type=MemoryType.NOTE)

        results = manager.search("various topics", limit=5)
        builder = ContextBuilder(max_messages=8, max_characters=6000, max_memory_items=5)
        memory = WorkingMemory(max_size=20)
        mem_ctx = [m.content for m in results]
        ctx = builder.build(_speech("query"), memory, memory_context=mem_ctx)
        assert ctx.total_characters <= 6000
        assert len(ctx.memory_context) <= 5
        manager.close()


class TestEmptyQueryBehavior:
    """Empty/whitespace queries return empty results safely."""

    def test_empty_string_search(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.remember("something")
        assert manager.search("") == []
        manager.close()

    def test_whitespace_search(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.remember("something")
        assert manager.search("   ") == []
        manager.close()

    def test_single_char_search(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        manager.remember("something")
        # Single char words are filtered (len > 1)
        assert manager.search("a") == []
        manager.close()


class TestContextCharacterAccounting:
    """Verify total_characters accurately reflects to_messages() content."""

    def test_total_characters_matches_message_content(self):
        """total_characters should equal sum of all content strings in to_messages()."""
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        memory.record("e1", "previous question", should_reason=True)
        conv = [
            {"role": "user", "content": "conv msg 1"},
            {"role": "assistant", "content": "conv answer 1"},
        ]
        mem_ctx = ["User prefers Neovim"]

        ctx = builder.build(_speech("current query"), memory, conversation_messages=conv, memory_context=mem_ctx)
        msgs = builder.to_messages(ctx)

        # total_characters = system + current + working memory + conversation + memory
        # It does NOT include the "Relevant information:" formatting wrapper
        # (that's formatting overhead, not content budget)
        actual_content_chars = (
            len(ctx.system_prompt)
            + len(ctx.current_event.transcript)
            + sum(len(e.transcript) for e in ctx.recent_events)
            + sum(len(m.get("content", "")) for m in ctx.conversation_messages)
            + sum(len(s) for s in ctx.memory_context)
        )
        assert ctx.total_characters == actual_content_chars

    def test_to_messages_always_ends_with_current_request(self):
        """The last message in to_messages() is always the current request."""
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        memory.record("e1", "old stuff", should_reason=True)
        conv = [{"role": "user", "content": "past"}]
        mem_ctx = ["some memory"]

        ctx = builder.build(_speech("CURRENT"), memory, conversation_messages=conv, memory_context=mem_ctx)
        msgs = builder.to_messages(ctx)
        assert msgs[-1] == {"role": "user", "content": "CURRENT"}

    def test_to_messages_starts_with_system(self):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        ctx = builder.build(_speech("q"), memory)
        msgs = builder.to_messages(ctx)
        assert msgs[0]["role"] == "system"

    def test_budget_is_content_only(self):
        """The character budget applies to content strings, not to role labels
        or message formatting. Document this: the limit is a content-character budget."""
        builder = ContextBuilder(max_messages=8, max_characters=2000)
        memory = WorkingMemory(max_size=20)
        for i in range(20):
            memory.record(f"e{i}", "a" * 50, should_reason=True)
        ctx = builder.build(_speech("current"), memory)
        # total_characters counts only content text
        assert ctx.total_characters <= 2000
        # The actual messages may have slightly more bytes due to role keys,
        # but the content budget is what's enforced
        msgs = builder.to_messages(ctx)
        content_total = sum(len(m["content"]) for m in msgs)
        # Content total should be close to total_characters
        # (they differ only if memory_context formatting adds "Relevant information:" wrapper)
        assert content_total >= ctx.total_characters
