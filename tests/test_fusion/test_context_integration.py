"""Integration tests: fusion → context builder → LLM messages."""

from __future__ import annotations

import time

import pytest

from app.config import FusionConfig
from app.context.builder import ContextBuilder
from app.events.types import Modality, SpeechEvent, VisualEvent
from app.fusion.engine import MultimodalFusion
from app.memory.working import WorkingMemory


def _speech(transcript: str, ts: float = 0.0) -> SpeechEvent:
    return SpeechEvent(
        timestamp=ts or time.time(),
        source=Modality.AUDIO,
        transcript=transcript,
    )


def _visual(description: str, ts: float = 0.0) -> VisualEvent:
    return VisualEvent(
        timestamp=ts or time.time(),
        source=Modality.VISUAL,
        description=description,
        event_type="SCENE_CHANGE",
        confidence=0.9,
    )


class TestFusionContextIntegration:
    """Verify fused context reaches the reasoning model correctly."""

    def test_speech_plus_visual_in_context(self):
        """When both speech and visual are temporally relevant, both appear in context."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        fusion.ingest(_visual("Python traceback visible.", ts=now))
        fusion.ingest(_speech("What is wrong with this?", ts=now + 2.0))

        snap = fusion.snapshot(reference_time=now + 2.0)
        vis_ctx = list(snap.visual_observations)

        builder = ContextBuilder(
            max_characters=6000,
            max_visual_events=5,
            max_visual_context_characters=500,
        )
        wm = WorkingMemory(max_size=10)
        event = _speech("What is wrong with this?", ts=now + 2.0)

        ctx = builder.build(event, wm, visual_context=vis_ctx)
        messages = builder.to_messages(ctx)

        # Visual context should be in messages
        visual_msgs = [m for m in messages if "visual context" in m.get("content", "").lower()]
        assert len(visual_msgs) == 1
        assert "Python traceback visible." in visual_msgs[0]["content"]

    def test_no_visual_when_expired(self):
        """When visual event has expired, context has no visual block."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        fusion.ingest(_visual("Old scene", ts=now - 30.0))

        snap = fusion.snapshot(reference_time=now)
        vis_ctx = list(snap.visual_observations) or None

        builder = ContextBuilder(max_characters=6000)
        wm = WorkingMemory(max_size=10)
        event = _speech("What is wrong with this?", ts=now)

        ctx = builder.build(event, wm, visual_context=vis_ctx)
        messages = builder.to_messages(ctx)

        visual_msgs = [m for m in messages if "visual context" in m.get("content", "").lower()]
        assert len(visual_msgs) == 0

    def test_audio_only_unchanged(self):
        """When fusion has no visual observations, behavior matches pre-Phase 8."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        fusion.ingest(_speech("Just a question", ts=now))

        snap = fusion.snapshot(reference_time=now)
        vis_ctx = list(snap.visual_observations) or None

        builder = ContextBuilder(max_characters=6000)
        wm = WorkingMemory(max_size=10)
        event = _speech("Just a question", ts=now)

        ctx = builder.build(event, wm, visual_context=vis_ctx)
        messages = builder.to_messages(ctx)

        # No visual block
        visual_msgs = [m for m in messages if "visual context" in m.get("content", "").lower()]
        assert len(visual_msgs) == 0

    def test_context_bounded_by_chars(self):
        """Visual context from fusion respects ContextBuilder character limits."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=60.0, max_events=100, max_characters=10000,
        ))

        # Add many visual observations
        for i in range(20):
            fusion.ingest(_visual(f"Visual observation {i} " * 10, ts=now + i * 0.1))

        snap = fusion.snapshot(reference_time=now + 2.0)
        vis_ctx = list(snap.visual_observations)

        builder = ContextBuilder(
            max_characters=6000,
            max_visual_events=3,
            max_visual_context_characters=200,
        )
        wm = WorkingMemory(max_size=10)
        event = _speech("Describe what you see", ts=now + 2.0)

        ctx = builder.build(event, wm, visual_context=vis_ctx)
        # Visual context is bounded by builder limits, not fusion limits
        assert len(ctx.visual_context) <= 3
        vis_chars = sum(len(v) for v in ctx.visual_context)
        assert vis_chars <= 200


class TestAttentionIntegration:
    """Verify fusion does not bypass attention — tested via main.py wiring.

    The visual attention handler gates what enters fusion. We test that
    non-attention-worthy events don't enter fusion by simulating the
    main.py flow logic.
    """

    def test_fusion_receives_only_attention_worthy_visuals(self):
        """Simulates the gating: only attention-worthy visuals enter fusion."""
        from app.attention.engine import AttentionEngine

        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        engine = AttentionEngine(reason_threshold=0.4)

        # Low-confidence event — attention won't pass it
        low = _visual("Boring scene", ts=now)
        low.confidence = 0.1
        low.event_type = "OBJECT_DETECTED"
        decision = engine.evaluate_visual(low)
        if decision.should_reason:
            fusion.ingest(low)

        # High-confidence scene change — attention passes it
        high = _visual("Person entering room", ts=now + 1.0)
        high.confidence = 0.9
        high.event_type = "SCENE_CHANGE"
        decision = engine.evaluate_visual(high)
        if decision.should_reason:
            fusion.ingest(high)

        snap = fusion.snapshot(reference_time=now + 1.0)
        assert "Boring scene" not in snap.visual_observations
        assert "Person entering room" in snap.visual_observations
