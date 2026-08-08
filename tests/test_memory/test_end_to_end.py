"""Full fake end-to-end test for Phase 5 pipeline.

SpeechEvent → Attention → ConversationState → Memory Retrieval →
Context Builder → Fake LLM → ResponseDecision → Conversation persistence.
"""
import asyncio

import pytest

from app.attention.engine import AttentionEngine
from app.context.builder import ContextBuilder
from app.events.bus import EventBus
from app.events.types import Event, Modality, ResponseEvent, SpeechEvent
from app.memory.manager import MemoryManager
from app.memory.sqlite_store import SQLiteStore
from app.memory.types import MemoryType
from app.memory.working import WorkingMemory
from app.reasoning.model import ReasoningResult
from app.response.decision import decide


def _speech(text: str, event_id: str | None = None) -> SpeechEvent:
    e = SpeechEvent(
        session_id="test", source=Modality.AUDIO,
        transcript=text, duration_seconds=1.0,
    )
    if event_id:
        object.__setattr__(e, "id", event_id)
    return e


class FakeModel:
    def __init__(self):
        self.calls = []

    @property
    def model_name(self) -> str:
        return "fake"

    def generate(self, messages, *, max_tokens=256, temperature=0.2):
        self.calls.append(messages)
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return ReasoningResult(
            text=f"Response to: {user_msg}",
            model="fake",
            latency_ms=1.0,
            success=True,
        )


class TestFullPhase5Pipeline:
    """End-to-end: interaction persists, next request sees history."""

    def test_single_interaction_persists(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        model = FakeModel()

        # User speaks
        event = _speech("Hey Jarvis, what time is it?")
        decision = engine.evaluate(event)
        assert decision.should_reason is True

        # Persist user message
        conv.add_user_message(event.transcript, source_event_id=event.id)

        # Build context with conversation
        recent = conv.recent_messages(limit=8)
        conv_msgs = [
            {"role": m.role, "content": m.content}
            for m in recent if m.source_event_id != event.id
        ]
        ctx = builder.build(event, engine.working_memory, conversation_messages=conv_msgs)
        messages = builder.to_messages(ctx)
        result = model.generate(messages)
        resp = decide(event.id, result)

        assert resp.respond is True

        # Persist assistant response
        conv.add_assistant_message(resp.text, source_event_id=event.id + "_response")

        # Verify persistence
        all_msgs = conv.recent_messages()
        assert len(all_msgs) == 2
        assert all_msgs[0].role == "user"
        assert all_msgs[1].role == "assistant"

        manager.close()

    def test_conversation_available_in_next_interaction(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        model = FakeModel()

        # First interaction
        event1 = _speech("Hey Jarvis, what is Python?")
        engine.evaluate(event1)
        conv.add_user_message(event1.transcript, source_event_id=event1.id)
        ctx1 = builder.build(event1, engine.working_memory)
        result1 = model.generate(builder.to_messages(ctx1))
        resp1 = decide(event1.id, result1)
        conv.add_assistant_message(resp1.text, source_event_id=event1.id + "_resp")

        # Second interaction — should see first in context
        event2 = _speech("Hey Jarvis, and what about Go?")
        engine.evaluate(event2)
        conv.add_user_message(event2.transcript, source_event_id=event2.id)

        recent = conv.recent_messages(limit=8)
        conv_msgs = [
            {"role": m.role, "content": m.content}
            for m in recent if m.source_event_id != event2.id
        ]
        ctx2 = builder.build(event2, engine.working_memory, conversation_messages=conv_msgs)
        messages2 = builder.to_messages(ctx2)

        # Verify previous conversation in context
        all_content = [m["content"] for m in messages2]
        assert any("Python" in c for c in all_content)

        manager.close()

    def test_memory_retrieval_in_context(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        model = FakeModel()

        # Store a preference
        manager.remember("User prefers Neovim editor", memory_type=MemoryType.PREFERENCE, importance=0.9)

        # Ask about editor
        event = _speech("Hey Jarvis, what editor should I use?")
        engine.evaluate(event)
        conv.add_user_message(event.transcript, source_event_id=event.id)

        # Retrieve relevant memories
        memories = manager.search(event.transcript, limit=5)
        mem_context = [m.content for m in memories] if memories else None

        ctx = builder.build(
            event, engine.working_memory,
            memory_context=mem_context,
        )
        messages = builder.to_messages(ctx)

        all_content = " ".join(m["content"] for m in messages)
        assert "Neovim" in all_content

        manager.close()

    def test_failed_reasoning_not_persisted(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)

        event = _speech("Hey Jarvis, help me")
        engine.evaluate(event)
        conv.add_user_message(event.transcript, source_event_id=event.id)

        # Model fails
        result = ReasoningResult(text="", model="fake", latency_ms=1.0, success=False, error="timeout")
        resp = decide(event.id, result)
        assert resp.respond is False

        # Do NOT persist assistant message
        all_msgs = conv.recent_messages()
        assert len(all_msgs) == 1
        assert all_msgs[0].role == "user"

        manager.close()

    def test_empty_response_not_persisted(self):
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()

        event = _speech("Hey Jarvis, test")
        conv.add_user_message(event.transcript, source_event_id=event.id)

        result = ReasoningResult(text="   ", model="fake", latency_ms=1.0, success=True)
        resp = decide(event.id, result)
        assert resp.respond is False

        all_msgs = conv.recent_messages()
        assert len(all_msgs) == 1
        assert all_msgs[0].role == "user"

        manager.close()

    def test_attention_negative_not_persisted(self):
        """Background utterances should not become conversation messages."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)

        event = _speech("I'll call John tomorrow.")
        decision = engine.evaluate(event)
        assert decision.should_reason is False

        # Do not persist
        all_msgs = conv.recent_messages()
        assert len(all_msgs) == 0

        manager.close()

    def test_duplicate_event_idempotent(self):
        """Same event processed twice should not create duplicate messages."""
        store = SQLiteStore(":memory:")
        manager = MemoryManager(store)
        conv = manager.start_session()

        event = _speech("Hey Jarvis, hello")
        conv.add_user_message(event.transcript, source_event_id=event.id)
        conv.add_user_message(event.transcript, source_event_id=event.id)

        all_msgs = conv.recent_messages()
        assert len(all_msgs) == 1

        manager.close()
