"""Tests for Phase 5 context builder integration with persistent memory."""
import pytest

from app.context.builder import ContextBuilder
from app.events.types import Modality, SpeechEvent
from app.memory.working import WorkingMemory


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(
        session_id="test", source=Modality.AUDIO,
        transcript=text, duration_seconds=1.0,
    )


class TestContextWithConversation:
    """Context builder integrates persistent conversation messages."""

    def test_conversation_messages_included(self):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        conv = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        ctx = builder.build(_speech("Tell me more"), memory, conversation_messages=conv)
        msgs = builder.to_messages(ctx)
        contents = [m["content"] for m in msgs]
        assert "What is Python?" in contents
        assert "Python is a programming language." in contents
        assert "Tell me more" in contents

    def test_conversation_roles_preserved(self):
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
        roles = [(m["role"], m["content"]) for m in msgs if m["role"] != "system"]
        assert ("user", "Q1") in roles
        assert ("assistant", "A1") in roles
        assert ("user", "Q3") in roles

    def test_current_request_always_present_with_conversation(self):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        conv = [{"role": "user", "content": "x" * 5000}]
        ctx = builder.build(_speech("current question"), memory, conversation_messages=conv)
        msgs = builder.to_messages(ctx)
        assert msgs[-1]["content"] == "current question"


class TestContextWithMemory:
    """Context builder integrates relevant persistent memories."""

    def test_memory_context_included(self):
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=20)
        mem_ctx = ["User prefers Neovim", "User works on Python projects"]
        ctx = builder.build(_speech("What editor?"), memory, memory_context=mem_ctx)
        msgs = builder.to_messages(ctx)
        combined = " ".join(m["content"] for m in msgs)
        assert "Neovim" in combined
        assert "Python projects" in combined

    def test_memory_bounded_by_max_items(self):
        builder = ContextBuilder(max_messages=8, max_characters=6000, max_memory_items=2)
        memory = WorkingMemory(max_size=20)
        mem_ctx = ["mem1", "mem2", "mem3", "mem4", "mem5"]
        ctx = builder.build(_speech("q"), memory, memory_context=mem_ctx)
        assert len(ctx.memory_context) <= 2

    def test_current_request_always_present_with_large_memory(self):
        builder = ContextBuilder(max_messages=8, max_characters=2000)
        memory = WorkingMemory(max_size=20)
        mem_ctx = ["x" * 500 for _ in range(10)]
        ctx = builder.build(_speech("current"), memory, memory_context=mem_ctx)
        msgs = builder.to_messages(ctx)
        assert msgs[-1]["content"] == "current"


class TestContextBoundsWithPhase5:
    """Context character/message limits still hold with conversation + memory."""

    def test_character_limit_with_all_sources(self):
        builder = ContextBuilder(max_messages=8, max_characters=2000)
        memory = WorkingMemory(max_size=20)
        for i in range(10):
            memory.record(f"e{i}", "a" * 50, should_reason=True)
        conv = [{"role": "user", "content": "b" * 50} for _ in range(10)]
        mem_ctx = ["c" * 50 for _ in range(10)]

        ctx = builder.build(_speech("current"), memory, conversation_messages=conv, memory_context=mem_ctx)
        assert ctx.total_characters <= 2000

    @pytest.mark.parametrize("msg_count", [10, 100, 1000, 10000])
    def test_context_bounded_at_scale(self, msg_count):
        """Even with thousands of messages, context stays bounded."""
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        memory = WorkingMemory(max_size=100)
        conv = [{"role": "user", "content": f"msg {i}"} for i in range(msg_count)]
        ctx = builder.build(_speech("current"), memory, conversation_messages=conv)
        assert ctx.total_characters <= 6000

    def test_irrelevant_memories_excluded_from_context(self):
        """Only memories that fit budget are included."""
        budget = len(ContextBuilder().system_prompt) + 340
        builder = ContextBuilder(max_messages=8, max_characters=budget, max_memory_items=5)
        memory = WorkingMemory(max_size=20)
        mem_ctx = ["x" * 200 for _ in range(5)]
        ctx = builder.build(_speech("current"), memory, memory_context=mem_ctx)
        # Budget constrains how many memories can be included
        assert ctx.total_characters <= budget
        assert len(ctx.memory_context) < 5


class TestBackwardsCompatibility:
    """Phase 5 context builder remains compatible with Phase 1-4 usage."""

    def test_build_without_conversation_or_memory(self):
        builder = ContextBuilder(max_messages=4, max_characters=2000)
        memory = WorkingMemory(max_size=20)
        memory.record("e1", "previous", should_reason=True)
        ctx = builder.build(_speech("current"), memory)
        msgs = builder.to_messages(ctx)
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["content"] == "current"
        # Still works exactly as Phase 3/4 expected
        assert all(m["role"] in ("system", "user") for m in msgs)
