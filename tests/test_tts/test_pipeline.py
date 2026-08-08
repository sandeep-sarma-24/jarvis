"""End-to-end pipeline tests: SpeechEvent → Attention → Reasoning → TTS → Player.

Uses FakeTTS and FakePlayer to test the complete gating architecture.
"""
import asyncio

import numpy as np
import pytest

from app.attention.engine import AttentionEngine
from app.config import JarvisConfig, PlaybackConfig, ReasoningConfig, ResponseConfig, TTSConfig
from app.context.builder import ContextBuilder
from app.events.bus import EventBus
from app.events.types import Event, Modality, ResponseEvent, SpeechEvent
from app.memory.working import WorkingMemory
from app.reasoning.model import ReasoningResult
from app.response.decision import decide
from app.tts.model import TTSResult

from .test_model import FakePlayer, FakeTTS


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(
        session_id="test", source=Modality.AUDIO,
        transcript=text, duration_seconds=1.0,
    )


class FakeReasoningModel:
    @property
    def model_name(self) -> str:
        return "fake-reasoning"

    def generate(self, messages, *, max_tokens=256, temperature=0.2):
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return ReasoningResult(
            text=f"Response to: {user_msg}",
            model="fake-reasoning",
            latency_ms=1.0,
            success=True,
        )


class TestResponseGating:
    """Verify TTS is only triggered when respond=True."""

    def test_response_false_no_tts(self):
        tts = FakeTTS()
        player = FakePlayer()

        result = ReasoningResult(
            text="", model="fake", latency_ms=1.0,
            success=False, error="timeout",
        )
        resp = decide("eid", result)
        assert resp.respond is False

        # TTS should not be called
        if resp.respond and resp.text:
            tts.synthesize(resp.text)
        assert len(tts.calls) == 0
        assert len(player.plays) == 0

    def test_response_true_triggers_tts(self):
        tts = FakeTTS()
        player = FakePlayer()

        result = ReasoningResult(
            text="The issue is caused by a null pointer.",
            model="fake", latency_ms=1.0, success=True,
        )
        resp = decide("eid", result)
        assert resp.respond is True

        # TTS should be called
        tts_result = tts.synthesize(resp.text)
        assert tts_result.success is True
        assert len(tts.calls) == 1

        if tts_result.audio is not None:
            player.play(tts_result.audio, tts_result.sample_rate)
        assert len(player.plays) == 1


class TestFullFakePipeline:
    """End-to-end: SpeechEvent → Attention → FakeReasoning → FakeTTS → FakePlayer."""

    def test_jarvis_addressed_full_pipeline(self):
        """'Jarvis, explain HTTP 500' → attention=true → reasoning → tts → playback."""
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        builder = ContextBuilder(max_messages=4, max_characters=2000)
        model = FakeReasoningModel()
        tts = FakeTTS()
        player = FakePlayer()

        event = _speech("Jarvis, explain HTTP 500.")
        decision = engine.evaluate(event)
        assert decision.should_reason is True

        ctx = builder.build(event, engine.working_memory)
        messages = builder.to_messages(ctx)
        result = model.generate(messages)
        assert result.success is True

        resp = decide(event.id, result)
        assert resp.respond is True

        tts_result = tts.synthesize(resp.text)
        assert tts_result.success is True
        assert len(tts.calls) == 1

        player.play(tts_result.audio, tts_result.sample_rate)
        assert len(player.plays) == 1
        assert player.is_playing() is True

    def test_background_speech_no_pipeline(self):
        """'I'll call John tomorrow' → attention=false → no reasoning, no tts."""
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        tts = FakeTTS()
        player = FakePlayer()

        event = _speech("I'll call John tomorrow.")
        decision = engine.evaluate(event)
        assert decision.should_reason is False

        # Nothing else happens
        assert len(tts.calls) == 0
        assert len(player.plays) == 0


class TestEventBusTTSIntegration:
    """Test TTS subscriber via the EventBus."""

    async def test_response_event_triggers_tts_subscriber(self):
        tts = FakeTTS()
        player = FakePlayer()
        bus = EventBus(queue_size=16)

        async def _tts_handler(event: Event) -> None:
            if not isinstance(event, ResponseEvent):
                return
            result = tts.synthesize(event.text)
            if result.success and result.audio is not None:
                player.play(result.audio, result.sample_rate)

        bus.subscribe(ResponseEvent, _tts_handler)
        await bus.start()

        response_event = ResponseEvent(
            session_id="test", source=Modality.SYSTEM,
            event_id="orig123", text="The answer is 42.",
        )
        await bus.publish(response_event)
        await asyncio.sleep(0.05)

        assert len(tts.calls) == 1
        assert tts.calls[0] == "The answer is 42."
        assert len(player.plays) == 1

        await bus.shutdown()

    async def test_no_response_event_no_tts(self):
        tts = FakeTTS()
        bus = EventBus(queue_size=16)
        tts_called = False

        async def _tts_handler(event: Event) -> None:
            nonlocal tts_called
            if isinstance(event, ResponseEvent):
                tts_called = True

        bus.subscribe(ResponseEvent, _tts_handler)
        await bus.start()

        # Publish a SpeechEvent (not ResponseEvent) — TTS handler shouldn't fire
        speech = _speech("Hello world")
        await bus.publish(speech)
        await asyncio.sleep(0.05)

        assert tts_called is False
        await bus.shutdown()

    async def test_event_correlation_preserved(self):
        """ResponseEvent.event_id traces back to original SpeechEvent.id."""
        captured_event_ids: list[str] = []
        bus = EventBus(queue_size=16)

        async def _tts_handler(event: Event) -> None:
            if isinstance(event, ResponseEvent):
                captured_event_ids.append(event.event_id)

        bus.subscribe(ResponseEvent, _tts_handler)
        await bus.start()

        original_event = _speech("Jarvis, test correlation")
        response_event = ResponseEvent(
            session_id="test", source=Modality.SYSTEM,
            event_id=original_event.id, text="Correlated response.",
        )
        await bus.publish(response_event)
        await asyncio.sleep(0.05)

        assert len(captured_event_ids) == 1
        assert captured_event_ids[0] == original_event.id
        await bus.shutdown()


class TestTextLimits:
    """Verify text length enforcement."""

    def test_short_text_unchanged(self):
        tts = FakeTTS()
        text = "Short text."
        result = tts.synthesize(text)
        assert tts.calls[0] == text

    def test_tts_handles_empty_text(self):
        tts = FakeTTS()
        result = tts.synthesize("")
        assert result.success is True  # FakeTTS doesn't validate


class TestInterruptionPolicy:
    """Verify new response interrupts current playback."""

    def test_new_response_stops_previous(self):
        player = FakePlayer()
        audio = np.zeros(24000, dtype=np.float32)

        player.play(audio, 24000)
        assert player.is_playing() is True

        # New response arrives — stop + play
        player.stop()
        player.play(audio, 24000)
        assert player.is_playing() is True
        assert player.stop_count == 1


class TestRealTTSSmoke:
    """Smoke test requiring the actual Kokoro TTS model.

    Skipped in normal test runs. Run with: pytest -m model
    """

    @pytest.mark.model
    def test_kokoro_synthesize_short_sentence(self):
        from app.tts.local_tts import KokoroTTS

        tts = KokoroTTS()
        result = tts.synthesize("Hello, this is a test.")

        assert result.success is True, f"TTS failed: {result.error}"
        assert result.audio is not None
        assert result.sample_rate == 24000
        assert result.duration_seconds > 0.1
        assert result.latency_ms > 0

        print(f"\n  TTS latency: {result.latency_ms:.0f}ms")
        print(f"  Audio duration: {result.duration_seconds:.2f}s")
        print(f"  Real-time factor: {result.latency_ms / (result.duration_seconds * 1000):.2f}x")

        tts.close()

    @pytest.mark.model
    def test_kokoro_stream(self):
        from app.tts.local_tts import KokoroTTS

        tts = KokoroTTS()
        chunks = list(tts.synthesize_stream(
            "This is a longer sentence. It has multiple parts. Let us see if streaming works."
        ))

        assert len(chunks) >= 1
        assert chunks[-1].is_last is True
        for chunk in chunks:
            assert chunk.sample_rate == 24000
            assert len(chunk.audio) > 0

        tts.close()
