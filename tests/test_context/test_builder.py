"""Tests for ContextBuilder."""
import time

import pytest

from app.context.builder import Context, ContextBuilder
from app.events.types import Modality, SpeechEvent
from app.memory.working import WorkingMemory


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(
        session_id="test", source=Modality.AUDIO,
        transcript=text, duration_seconds=1.0,
    )


@pytest.fixture
def builder():
    return ContextBuilder(max_messages=4, max_characters=2000)


@pytest.fixture
def memory():
    return WorkingMemory(max_size=20)


class TestCurrentEventAlwaysPresent:

    def test_empty_history(self, builder, memory):
        ctx = builder.build(_speech("What is 2+2?"), memory)
        assert ctx.current_event.transcript == "What is 2+2?"
        assert len(ctx.recent_events) == 0

    def test_with_history(self, builder, memory):
        memory.record("e1", "Hey Jarvis", should_reason=True)
        memory.record("e2", "background noise", should_reason=False)
        ctx = builder.build(_speech("How does this work?"), memory)
        assert ctx.current_event.transcript == "How does this work?"

    def test_current_event_not_truncated(self, builder, memory):
        long_text = "x" * 5000
        ctx = builder.build(_speech(long_text), memory)
        assert ctx.current_event.transcript == long_text

    def test_large_current_event_with_large_history(self):
        """Current event exceeding the historical budget still appears."""
        builder = ContextBuilder(max_messages=4, max_characters=2000)
        memory = WorkingMemory(max_size=20)
        for i in range(10):
            memory.record(f"e{i}", "a" * 100, should_reason=True)
        big = _speech("z" * 3000)
        ctx = builder.build(big, memory)
        assert ctx.current_event.transcript == "z" * 3000
        assert len(ctx.recent_events) == 0


class TestCurrentEventNeverDuplicated:
    """The current event must appear exactly once in the model context,
    even when it already exists in working memory."""

    def test_no_duplication_after_attention(self):
        from app.attention.engine import AttentionEngine
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=8, max_characters=6000)

        event = _speech("Jarvis, what is the weather?")
        engine.evaluate(event)

        ctx = builder.build(event, engine.working_memory)
        msgs = builder.to_messages(ctx)
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_contents.count("Jarvis, what is the weather?") == 1

    def test_no_duplication_manual_record(self, builder):
        memory = WorkingMemory(max_size=20)
        event = _speech("test query")
        memory.record(event.id, event.transcript, should_reason=True)

        ctx = builder.build(event, memory)
        msgs = builder.to_messages(ctx)
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_contents.count("test query") == 1

    def test_historical_events_still_included(self):
        from app.attention.engine import AttentionEngine
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=8, max_characters=6000)

        engine.evaluate(_speech("Hey Jarvis, what is Python?"))
        event2 = _speech("Hey Jarvis, and what about Go?")
        engine.evaluate(event2)

        ctx = builder.build(event2, engine.working_memory)
        msgs = builder.to_messages(ctx)
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert "Hey Jarvis, what is Python?" in user_contents
        assert user_contents.count("Hey Jarvis, and what about Go?") == 1


class TestRecentEventsBounded:

    def test_max_messages_enforced(self, memory):
        builder = ContextBuilder(max_messages=3, max_characters=10000)
        for i in range(10):
            memory.record(f"e{i}", f"text {i}", should_reason=True)
        ctx = builder.build(_speech("current"), memory)
        assert len(ctx.recent_events) <= 3

    def test_prefers_should_reason_entries(self, memory):
        builder = ContextBuilder(max_messages=3, max_characters=10000)
        memory.record("bg1", "background 1", should_reason=False)
        memory.record("int1", "interaction 1", should_reason=True)
        memory.record("bg2", "background 2", should_reason=False)
        memory.record("int2", "interaction 2", should_reason=True)
        ctx = builder.build(_speech("current"), memory)
        transcripts = [e.transcript for e in ctx.recent_events]
        assert "interaction 1" in transcripts
        assert "interaction 2" in transcripts

    def test_selection_excludes_current_event(self):
        """Historical selection must not include the current event."""
        builder = ContextBuilder(max_messages=10, max_characters=10000)
        memory = WorkingMemory(max_size=20)
        memory.record("hist1", "historical meaningful", should_reason=True)
        memory.record("hist2", "historical background", should_reason=False)
        event = _speech("current request")
        memory.record(event.id, event.transcript, should_reason=True)
        ctx = builder.build(event, memory)
        hist_ids = [e.event_id for e in ctx.recent_events]
        assert event.id not in hist_ids
        assert "hist1" in hist_ids

    def test_ordering_is_chronological(self, memory):
        builder = ContextBuilder(max_messages=10, max_characters=10000)
        memory.record("e1", "first", should_reason=True)
        memory.record("e2", "second", should_reason=True)
        memory.record("e3", "third", should_reason=True)
        ctx = builder.build(_speech("current"), memory)
        transcripts = [e.transcript for e in ctx.recent_events]
        assert transcripts == ["first", "second", "third"]

    def test_messages_are_older_then_newer_then_current(self, memory):
        builder = ContextBuilder(max_messages=10, max_characters=10000)
        memory.record("e1", "older interaction", should_reason=True)
        memory.record("e2", "newer interaction", should_reason=True)
        ctx = builder.build(_speech("current event"), memory)
        msgs = builder.to_messages(ctx)
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_msgs == ["older interaction", "newer interaction", "current event"]


class TestMaxCharactersRespected:

    def test_character_budget_enforced(self, memory):
        builder = ContextBuilder(max_messages=100, max_characters=2000)
        for i in range(50):
            memory.record(f"e{i}", "a" * 100, should_reason=True)
        ctx = builder.build(_speech("current"), memory)
        assert ctx.total_characters <= 2000

    def test_system_prompt_counted(self, memory):
        builder = ContextBuilder(max_messages=10, max_characters=500)
        ctx = builder.build(_speech("current"), memory)
        assert ctx.total_characters >= len(builder.system_prompt)


class TestEmptyHistory:

    def test_no_recent_events(self, builder, memory):
        ctx = builder.build(_speech("hello"), memory)
        assert ctx.recent_events == ()
        assert ctx.current_event.transcript == "hello"


class TestToMessages:

    def test_message_format(self, builder, memory):
        memory.record("e1", "previous question", should_reason=True)
        ctx = builder.build(_speech("current question"), memory)
        msgs = builder.to_messages(ctx)
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "current question"
        assert len(msgs) >= 2

    def test_empty_history_messages(self, builder, memory):
        ctx = builder.build(_speech("just me"), memory)
        msgs = builder.to_messages(ctx)
        assert len(msgs) == 2  # system + current
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "just me"

    def test_all_history_entries_are_user_role(self, builder, memory):
        memory.record("e1", "question 1", should_reason=True)
        memory.record("e2", "background", should_reason=False)
        ctx = builder.build(_speech("current"), memory)
        msgs = builder.to_messages(ctx)
        for m in msgs[1:]:
            assert m["role"] == "user"


class TestContextGrowthBounded:
    """Simulate increasing event counts and verify context stays bounded."""

    @pytest.mark.parametrize("event_count", [1, 5, 10, 20, 50, 100])
    def test_context_bounded_at_scale(self, event_count):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=100)
        for i in range(event_count):
            memory.record(f"e{i}", f"event text number {i} " * 5, should_reason=True)
        ctx = builder.build(_speech("current query"), memory)
        assert ctx.total_characters <= 6000
        assert len(ctx.recent_events) <= 8


class TestConversationHistory:

    def test_keeps_newest_turns_over_budget(self, memory):
        builder = ContextBuilder(max_messages=8, max_characters=10_000)
        budget = 10_000 - len(builder.system_prompt) - len("current")
        turns = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"{i:03d}" + "x" * 97}
            for i in range(budget // 100 + 10)
        ]
        ctx = builder.build(_speech("current"), memory, conversation_messages=turns)
        kept = list(ctx.conversation_messages)
        assert kept and len(kept) < len(turns)
        assert kept == turns[-len(kept):]  # newest suffix, chronological
        assert ctx.total_characters <= 10_000

    def test_history_utterance_not_repeated_from_working_memory(self, memory):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory.record("e1", "what is the capital of France?", should_reason=True)
        memory.record("e2", "someone talking in the background", should_reason=False)
        history = [
            {"role": "user", "content": "what is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
        event = _speech("and of Spain?")
        memory.record(event.id, event.transcript, should_reason=True)
        msgs = builder.to_messages(builder.build(event, memory, conversation_messages=history))
        contents = [m["content"] for m in msgs[1:]]
        assert contents.count("what is the capital of France?") == 1
        # Question, its answer, other overheard speech, then the current turn.
        assert contents == [
            "what is the capital of France?", "Paris.",
            "someone talking in the background", "and of Spain?",
        ]
