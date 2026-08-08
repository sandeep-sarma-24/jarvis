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

    def test_ordering_is_chronological(self, memory):
        builder = ContextBuilder(max_messages=10, max_characters=10000)
        memory.record("e1", "first", should_reason=True)
        memory.record("e2", "second", should_reason=True)
        memory.record("e3", "third", should_reason=True)
        ctx = builder.build(_speech("current"), memory)
        transcripts = [e.transcript for e in ctx.recent_events]
        assert transcripts == ["first", "second", "third"]


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
