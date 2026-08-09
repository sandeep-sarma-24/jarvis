"""Integration tests: fusion → context builder → LLM messages.

Tests the corrected Phase 8 architecture where fusion receives events
independently of attention, and multimodal context (not just visual)
reaches the ContextBuilder.
"""

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


def _mm_ctx_from_snap(snap, event_id: str | None = None):
    """Format multimodal context from snapshot, as main.py does."""
    parts = []
    for obs in snap.audio_observations:
        parts.append(f"[audio] {obs}")
    for obs in snap.visual_observations:
        parts.append(f"[visual] {obs}")
    return parts if parts else None


class TestFusionContextIntegration:
    """Verify fused context reaches the reasoning model correctly."""

    def test_speech_plus_visual_in_context(self):
        """When both speech and visual are temporally relevant, both appear in context."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        visual = _visual("Python traceback visible.", ts=now)
        speech = _speech("What is wrong with this?", ts=now + 2.0)

        fusion.ingest(visual)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})
        mm_ctx = _mm_ctx_from_snap(snap)

        builder = ContextBuilder(
            max_characters=6000,
            max_visual_events=5,
            max_visual_context_characters=500,
        )
        wm = WorkingMemory(max_size=10)

        ctx = builder.build(speech, wm, multimodal_context=mm_ctx)
        messages = builder.to_messages(ctx)

        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 1
        assert "[visual] Python traceback visible." in obs_msgs[0]["content"]

    def test_no_visual_when_expired(self):
        """When visual event has expired, context has no observations block."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        fusion.ingest(_visual("Old scene", ts=now - 30.0))

        speech = _speech("What is wrong with this?", ts=now)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})
        mm_ctx = _mm_ctx_from_snap(snap)

        builder = ContextBuilder(max_characters=6000)
        wm = WorkingMemory(max_size=10)

        ctx = builder.build(speech, wm, multimodal_context=mm_ctx)
        messages = builder.to_messages(ctx)

        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 0

    def test_audio_only_unchanged(self):
        """When fusion has only the current speech (excluded), no multimodal block appears."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        speech = _speech("Just a question", ts=now)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})
        mm_ctx = _mm_ctx_from_snap(snap)

        builder = ContextBuilder(max_characters=6000)
        wm = WorkingMemory(max_size=10)

        ctx = builder.build(speech, wm, multimodal_context=mm_ctx)
        messages = builder.to_messages(ctx)

        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 0

    def test_context_bounded_by_chars(self):
        """Multimodal context from fusion respects ContextBuilder character limits."""
        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=60.0, max_events=100, max_characters=10000,
        ))

        for i in range(20):
            fusion.ingest(_visual(f"Visual observation {i} " * 10, ts=now + i * 0.1))

        speech = _speech("Describe what you see", ts=now + 2.0)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})
        mm_ctx = _mm_ctx_from_snap(snap)

        builder = ContextBuilder(
            max_characters=6000,
            max_visual_events=3,
            max_visual_context_characters=200,
        )
        wm = WorkingMemory(max_size=10)

        ctx = builder.build(speech, wm, multimodal_context=mm_ctx)
        # Multimodal context is bounded by builder limits
        assert len(ctx.multimodal_context) <= 3
        mm_chars = sum(len(s) for s in ctx.multimodal_context)
        assert mm_chars <= 200


class TestAttentionIntegration:
    """Verify that fusion is independent of attention in the corrected architecture."""

    def test_all_visuals_enter_fusion_regardless_of_attention(self):
        """Corrected architecture: ALL visual events enter fusion, not just attention-worthy ones."""
        from app.attention.engine import AttentionEngine

        now = time.time()
        fusion = MultimodalFusion(FusionConfig(
            enabled=True, window_seconds=10.0, max_events=20, max_characters=3000,
        ))

        engine = AttentionEngine(reason_threshold=0.4)

        # Low-confidence event — attention rejects it
        low = _visual("Boring scene", ts=now)
        low.confidence = 0.1
        low.event_type = "OBJECT_DETECTED"
        decision = engine.evaluate_visual(low)
        assert not decision.should_reason

        # In the corrected architecture, fusion receives it anyway
        fusion.ingest(low)

        # High-confidence scene change
        high = _visual("Person entering room", ts=now + 1.0)
        high.confidence = 0.9
        high.event_type = "SCENE_CHANGE"
        fusion.ingest(high)

        snap = fusion.snapshot(reference_time=now + 1.0)
        # Both events are in fusion — attention is not a gate for fusion
        assert "Boring scene" in snap.visual_observations
        assert "Person entering room" in snap.visual_observations
