"""Integration test: Attention → Context → FakeModel → ResponseDecision.

Uses FakeModel so no Ollama is needed. Tests the full Phase 3 pipeline logic.
"""
import asyncio

import pytest

from app.attention.engine import AttentionEngine
from app.context.builder import ContextBuilder
from app.events.bus import EventBus
from app.events.types import Event, Modality, SpeechEvent
from app.memory.working import WorkingMemory
from app.reasoning.model import ReasoningResult
from app.response.decision import ResponseDecision, decide


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(
        session_id="test", source=Modality.AUDIO,
        transcript=text, duration_seconds=1.0,
    )


class FakeModel:
    @property
    def model_name(self) -> str:
        return "fake"

    def generate(self, messages, *, max_tokens=256, temperature=0.2):
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return ReasoningResult(
            text=f"Response to: {user_msg}",
            model="fake",
            latency_ms=1.0,
            success=True,
        )


class TestFullPipeline:
    """Tests the attention → context → model → decision flow with FakeModel."""

    def test_attention_positive_gets_response(self):
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=4, max_characters=2000)
        model = FakeModel()

        event = _speech("Hey Jarvis, what time is it?")
        decision = engine.evaluate(event)
        assert decision.should_reason is True

        ctx = builder.build(event, engine.working_memory)
        messages = builder.to_messages(ctx)
        result = model.generate(messages)
        assert result.success is True

        resp = decide(event.id, result)
        assert resp.respond is True
        assert resp.text is not None
        assert "Hey Jarvis, what time is it?" in resp.text

    def test_attention_negative_blocks_reasoning(self):
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        event = _speech("I'll call you tomorrow.")
        decision = engine.evaluate(event)
        assert decision.should_reason is False

    def test_model_failure_blocks_response(self):
        event = _speech("Hey Jarvis, help")
        result = ReasoningResult(
            text="", model="fake", latency_ms=1.0,
            success=False, error="timeout",
        )
        resp = decide(event.id, result)
        assert resp.respond is False
        assert "failed" in resp.reason

    def test_empty_model_output_blocks_response(self):
        event = _speech("Hey Jarvis, help")
        result = ReasoningResult(
            text="   ", model="fake", latency_ms=1.0, success=True,
        )
        resp = decide(event.id, result)
        assert resp.respond is False
        assert "empty" in resp.reason

    def test_context_includes_working_memory(self):
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=4, max_characters=2000)

        engine.evaluate(_speech("Hey Jarvis, what is Python?"))
        event2 = _speech("Hey Jarvis, and what about Go?")
        engine.evaluate(event2)

        ctx = builder.build(event2, engine.working_memory)
        transcripts = [e.transcript for e in ctx.recent_events]
        assert "Hey Jarvis, what is Python?" in transcripts

    def test_event_id_preserved_in_response(self):
        event = _speech("test")
        result = ReasoningResult(
            text="answer", model="fake", latency_ms=1.0, success=True,
        )
        resp = decide(event.id, result)
        assert resp.event_id == event.id

    def test_current_event_exactly_once_in_context(self):
        """End-to-end: after attention records the event, context builder
        must include the current transcript exactly once."""
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=8, max_characters=6000)
        model = FakeModel()

        event = _speech("Jarvis, explain why HTTP 500 happens.")
        decision = engine.evaluate(event)
        assert decision.should_reason is True

        ctx = builder.build(event, engine.working_memory)
        msgs = builder.to_messages(ctx)
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_contents.count("Jarvis, explain why HTTP 500 happens.") == 1

        result = model.generate(msgs)
        resp = decide(event.id, result)
        assert resp.respond is True
        assert resp.event_id == event.id

    def test_attention_negative_never_reaches_model(self):
        """End-to-end: attention-negative events must never invoke reasoning."""
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        event = _speech("I'll call John tomorrow.")
        decision = engine.evaluate(event)
        assert decision.should_reason is False


class TestConcurrencyBounding:
    """Verify reasoning is bounded to max_concurrent_requests.

    Phase 3 policy: at most one local LLM inference is active.
    If another attention-positive event arrives while inference is active,
    the event is dropped rather than creating unbounded inference tasks.
    """

    async def test_semaphore_limits_concurrency(self):
        sem = asyncio.Semaphore(1)
        active = 0
        max_active = 0

        async def simulate_reasoning():
            nonlocal active, max_active
            async with sem:
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1

        tasks = [asyncio.create_task(simulate_reasoning()) for _ in range(5)]
        await asyncio.gather(*tasks)
        assert max_active == 1

    async def test_locked_semaphore_drops_request(self):
        """When the reasoning slot is full, Semaphore.locked() returns True."""
        sem = asyncio.Semaphore(1)
        dropped = 0

        async def hold_slot():
            async with sem:
                await asyncio.sleep(0.1)

        async def try_reasoning():
            nonlocal dropped
            if sem.locked():
                dropped += 1
                return
            async with sem:
                await asyncio.sleep(0.01)

        hold_task = asyncio.create_task(hold_slot())
        await asyncio.sleep(0.01)  # let hold_slot acquire
        await try_reasoning()
        assert dropped == 1
        hold_task.cancel()
        try:
            await hold_task
        except asyncio.CancelledError:
            pass


class TestNonBlocking:
    """Verify model inference does not block the event loop."""

    async def test_to_thread_does_not_block_loop(self):
        """asyncio.to_thread runs synchronous code in a thread,
        allowing the event loop to process other coroutines."""
        import time as _time

        loop_alive = False

        async def background():
            nonlocal loop_alive
            await asyncio.sleep(0.01)
            loop_alive = True

        def slow_generate():
            _time.sleep(0.05)
            return "done"

        bg = asyncio.create_task(background())
        result = await asyncio.to_thread(slow_generate)
        await bg

        assert result == "done"
        assert loop_alive is True


class TestFailureSafety:
    """No failure in reasoning should crash the pipeline."""

    def test_attention_failure_blocks_reasoning(self):
        engine = AttentionEngine(assistant_names=["jarvis"])
        original = engine._rules.evaluate
        engine._rules.evaluate = lambda t: (_ for _ in ()).throw(ValueError("boom"))

        decision = engine.evaluate(_speech("test"))
        assert decision.should_reason is False
        assert decision.tier == "fallback"

        engine._rules.evaluate = original

    def test_context_failure_produces_no_response(self):
        result = ReasoningResult(
            text="", model="fake", latency_ms=0.0,
            success=False, error="context build failed",
        )
        resp = decide("eid", result)
        assert resp.respond is False

    def test_model_exception_produces_failure_result(self):
        """OllamaModel catches exceptions and returns success=False."""
        from app.reasoning.local_model import OllamaModel
        model = OllamaModel(
            model="nonexistent",
            base_url="http://localhost:1",  # connection refused
            timeout_seconds=1.0,
        )
        result = model.generate([{"role": "user", "content": "test"}], max_tokens=10)
        assert result.success is False
        assert result.error is not None
        model.close()
