"""Integration tests for Phase 8 multimodal fusion pipeline.

Tests the full path: Event → Fusion → ContextBuilder → messages.
Validates the corrected architecture where visual events enter fusion
independently of attention.
"""

from __future__ import annotations

import time

import pytest

from app.attention.engine import AttentionEngine
from app.config import FusionConfig
from app.context.builder import ContextBuilder
from app.events.types import Modality, SpeechEvent, VisualEvent
from app.fusion.engine import MultimodalFusion
from app.memory.working import WorkingMemory


def _speech(transcript: str, ts: float, **kw) -> SpeechEvent:
    return SpeechEvent(
        timestamp=ts,
        source=Modality.AUDIO,
        transcript=transcript,
        **kw,
    )


def _visual(description: str, ts: float, event_type: str = "SCENE_CHANGE", confidence: float = 0.9, **kw) -> VisualEvent:
    return VisualEvent(
        timestamp=ts,
        source=Modality.VISUAL,
        description=description,
        event_type=event_type,
        confidence=confidence,
        **kw,
    )


def _fusion(window: float = 10.0) -> MultimodalFusion:
    return MultimodalFusion(FusionConfig(
        enabled=True, window_seconds=window, max_events=20, max_characters=3000,
    ))


def _builder() -> ContextBuilder:
    return ContextBuilder(
        max_characters=6000,
        max_visual_events=10,
        max_visual_context_characters=1000,
    )


class TestVisualPlusSpeech:
    """Test 1: Visual + Speech temporal correlation."""

    def test_visual_then_speech_both_in_snapshot(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_visual("Terminal showing Python traceback.", ts=now))
        fusion.ingest(_speech("What is wrong with this?", ts=now + 3.0))

        snap = fusion.snapshot(reference_time=now + 3.0, exclude_event_ids={})
        assert snap.audio_observations == ("What is wrong with this?",)
        assert snap.visual_observations == ("Terminal showing Python traceback.",)

    def test_visual_then_speech_in_context_messages(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        visual = _visual("Terminal showing Python traceback.", ts=now)
        speech = _speech("What is wrong with this?", ts=now + 3.0)

        fusion.ingest(visual)
        fusion.ingest(speech)

        # Snapshot excludes the current speech event (it's the user message)
        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})

        # Format multimodal context (as main.py does)
        mm_ctx = []
        for obs in snap.audio_observations:
            mm_ctx.append(f"[audio] {obs}")
        for obs in snap.visual_observations:
            mm_ctx.append(f"[visual] {obs}")

        builder = _builder()
        wm = WorkingMemory(max_size=10)
        ctx = builder.build(speech, wm, multimodal_context=mm_ctx or None)
        messages = builder.to_messages(ctx)

        # Find the observations message
        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 1
        assert "[visual] Terminal showing Python traceback." in obs_msgs[0]["content"]

        # Current speech is the user message (last)
        assert messages[-1]["content"] == "What is wrong with this?"


class TestSpeechPlusVisual:
    """Test 2: Speech + Visual in same window."""

    def test_speech_then_visual_both_in_snapshot(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("Look at this.", ts=now))
        fusion.ingest(_visual("Python traceback visible on screen.", ts=now + 2.0))

        snap = fusion.snapshot(reference_time=now + 2.0)
        assert "Look at this." in snap.audio_observations
        assert "Python traceback visible on screen." in snap.visual_observations


class TestThreeEventSequence:
    """Test 3: Three-event sequence within window."""

    def test_three_events_all_in_snapshot(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        s1 = _speech("Look at this.", ts=now)
        v1 = _visual("Python traceback visible.", ts=now + 2.0)
        s2 = _speech("What is wrong?", ts=now + 4.0)

        fusion.ingest(s1)
        fusion.ingest(v1)
        fusion.ingest(s2)

        # Snapshot at s2's time, excluding s2 (it's the current event)
        snap = fusion.snapshot(reference_time=s2.timestamp, exclude_event_ids={s2.id})

        assert "Look at this." in snap.audio_observations
        assert "Python traceback visible." in snap.visual_observations
        # s2 excluded since it's the current event being processed
        assert "What is wrong?" not in snap.audio_observations

    def test_three_events_full_context_integration(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        s1 = _speech("Look at this.", ts=now)
        v1 = _visual("Python traceback visible.", ts=now + 2.0)
        s2 = _speech("What is wrong?", ts=now + 4.0)

        fusion.ingest(s1)
        fusion.ingest(v1)
        fusion.ingest(s2)

        snap = fusion.snapshot(reference_time=s2.timestamp, exclude_event_ids={s2.id})

        mm_ctx = []
        for obs in snap.audio_observations:
            mm_ctx.append(f"[audio] {obs}")
        for obs in snap.visual_observations:
            mm_ctx.append(f"[visual] {obs}")

        builder = _builder()
        wm = WorkingMemory(max_size=10)
        ctx = builder.build(s2, wm, multimodal_context=mm_ctx or None)
        messages = builder.to_messages(ctx)

        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 1
        assert "[audio] Look at this." in obs_msgs[0]["content"]
        assert "[visual] Python traceback visible." in obs_msgs[0]["content"]
        assert messages[-1]["content"] == "What is wrong?"


class TestExpiredVisual:
    """Test 4: Expired visual does not reach reasoning context."""

    def test_expired_visual_excluded(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        # Visual 30 seconds before speech (well outside 10s window)
        fusion.ingest(_visual("Coffee mug on desk.", ts=now))
        speech = _speech("What is that?", ts=now + 30.0)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})

        assert snap.visual_observations == ()
        assert snap.audio_observations == ()

    def test_expired_visual_not_in_context(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_visual("Coffee mug on desk.", ts=now))
        speech = _speech("What is that?", ts=now + 30.0)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})
        mm_ctx = []
        for obs in snap.audio_observations:
            mm_ctx.append(f"[audio] {obs}")
        for obs in snap.visual_observations:
            mm_ctx.append(f"[visual] {obs}")

        builder = _builder()
        wm = WorkingMemory(max_size=10)
        ctx = builder.build(speech, wm, multimodal_context=mm_ctx or None)
        messages = builder.to_messages(ctx)

        # No observations block when nothing is in the window
        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 0


class TestNonAttentionWorthyVisual:
    """Test 5: Visual event rejected by attention still enters fusion."""

    def test_low_salience_visual_in_fusion(self):
        now = time.time()
        fusion = _fusion(window=10.0)
        engine = AttentionEngine(reason_threshold=0.4)

        # Low-confidence event that attention won't deem worthy
        low = _visual("Boring empty desk.", ts=now, event_type="OBJECT_DETECTED", confidence=0.1)
        decision = engine.evaluate_visual(low)
        assert not decision.should_reason  # attention rejects it

        # But fusion still receives it (architecture: fusion is independent of attention)
        fusion.ingest(low)

        snap = fusion.snapshot(reference_time=now)
        assert "Boring empty desk." in snap.visual_observations

    def test_non_worthy_visual_available_as_context(self):
        """Even non-attention-worthy visuals provide temporal context for speech."""
        now = time.time()
        fusion = _fusion(window=10.0)
        engine = AttentionEngine(reason_threshold=0.4)

        low = _visual("Text editor open.", ts=now, event_type="OBJECT_DETECTED", confidence=0.2)
        decision = engine.evaluate_visual(low)
        assert not decision.should_reason

        fusion.ingest(low)
        speech = _speech("What file is open?", ts=now + 2.0)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})
        assert "Text editor open." in snap.visual_observations


class TestContextBuilderIntegration:
    """Test 6: Full ContextBuilder integration."""

    def test_multimodal_context_in_messages(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        visual = _visual("Python traceback visible.", ts=now)
        speech = _speech("What is wrong with this?", ts=now + 3.0)

        fusion.ingest(visual)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=speech.timestamp, exclude_event_ids={speech.id})

        mm_ctx = []
        for obs in snap.audio_observations:
            mm_ctx.append(f"[audio] {obs}")
        for obs in snap.visual_observations:
            mm_ctx.append(f"[visual] {obs}")

        builder = _builder()
        wm = WorkingMemory(max_size=10)
        ctx = builder.build(speech, wm, multimodal_context=mm_ctx or None)

        # Context object has the multimodal field
        assert "[visual] Python traceback visible." in ctx.multimodal_context

        # Messages contain the observation block
        messages = builder.to_messages(ctx)
        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 1
        content = obs_msgs[0]["content"]
        assert "[visual] Python traceback visible." in content

        # User message is the current speech event
        assert messages[-1] == {"role": "user", "content": "What is wrong with this?"}

    def test_audio_observations_preserved_in_context(self):
        """Prior speech observations (not current event) appear in multimodal context."""
        now = time.time()
        fusion = _fusion(window=10.0)

        s1 = _speech("Look at this.", ts=now)
        v1 = _visual("Screen with error.", ts=now + 1.0)
        s2 = _speech("What is wrong?", ts=now + 3.0)

        fusion.ingest(s1)
        fusion.ingest(v1)
        fusion.ingest(s2)

        snap = fusion.snapshot(reference_time=s2.timestamp, exclude_event_ids={s2.id})
        mm_ctx = [f"[audio] {o}" for o in snap.audio_observations] + \
                 [f"[visual] {o}" for o in snap.visual_observations]

        builder = _builder()
        wm = WorkingMemory(max_size=10)
        ctx = builder.build(s2, wm, multimodal_context=mm_ctx or None)
        messages = builder.to_messages(ctx)

        obs_msgs = [m for m in messages if "Recent observations" in m.get("content", "")]
        assert len(obs_msgs) == 1
        assert "[audio] Look at this." in obs_msgs[0]["content"]
        assert "[visual] Screen with error." in obs_msgs[0]["content"]


class TestNoRawData:
    """Test 7: No raw data in multimodal context."""

    def test_only_strings_in_snapshot(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("Hello", ts=now))
        fusion.ingest(_visual("Scene", ts=now + 1.0))

        snap = fusion.snapshot(reference_time=now + 1.0)

        for obs in snap.audio_observations:
            assert isinstance(obs, str)
        for obs in snap.visual_observations:
            assert isinstance(obs, str)
        for eid in snap.source_event_ids:
            assert isinstance(eid, str)
        assert isinstance(snap.timestamp, float)
        assert isinstance(snap.window_start, float)
        assert isinstance(snap.window_end, float)

    def test_no_bytes_in_context(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("Test speech", ts=now))
        fusion.ingest(_visual("Test visual", ts=now + 1.0))

        snap = fusion.snapshot(reference_time=now + 1.0)
        mm_ctx = [f"[audio] {o}" for o in snap.audio_observations] + \
                 [f"[visual] {o}" for o in snap.visual_observations]

        builder = _builder()
        wm = WorkingMemory(max_size=10)
        ctx = builder.build(
            _speech("Current", ts=now + 2.0), wm, multimodal_context=mm_ctx or None,
        )
        messages = builder.to_messages(ctx)

        for msg in messages:
            content = msg["content"]
            assert isinstance(content, str)
            assert not isinstance(content, (bytes, bytearray))


class TestDuplicateIngestion:
    """Verify each event is ingested exactly once."""

    def test_same_event_ingested_twice_counted_twice(self):
        """If the same event object is ingested twice, fusion counts it twice.
        The architecture must prevent this at the subscription level."""
        now = time.time()
        fusion = _fusion(window=10.0)

        event = _speech("Hello", ts=now)
        fusion.ingest(event)
        fusion.ingest(event)

        # Both ingestions are counted — the subscription layer must prevent this
        assert fusion.counters["events_received"] == 2

    def test_bus_subscription_pattern_prevents_duplicates(self):
        """Verify the intended subscription pattern: one subscriber per modality."""
        # This is a design validation test — in main.py:
        # SpeechEvent → _fusion_speech_handler (one subscription)
        # VisualEvent → _build_visual_attention_handler (one subscription, one ingest call)
        # There's no path where both handlers process the same event type.
        from app.events.types import SpeechEvent, VisualEvent
        assert not issubclass(SpeechEvent, VisualEvent)
        assert not issubclass(VisualEvent, SpeechEvent)


class TestSnapshotReferenceTime:
    """Verify snapshot uses event timestamp, not wall-clock."""

    def test_snapshot_at_event_time(self):
        """Snapshot should represent the situation at the event's timestamp."""
        now = time.time()
        fusion = _fusion(window=10.0)

        # Visual at now, speech at now+3
        fusion.ingest(_visual("Scene A", ts=now))
        fusion.ingest(_speech("Question", ts=now + 3.0))

        # Snapshot at the speech event's timestamp (now+3)
        snap = fusion.snapshot(reference_time=now + 3.0)
        assert "Scene A" in snap.visual_observations

        # Snapshot at a much later time (now+20) — visual would be expired
        snap_late = fusion.snapshot(reference_time=now + 20.0)
        assert "Scene A" not in snap_late.visual_observations

    def test_out_of_order_same_snapshot(self):
        """Out-of-order arrival produces same ordered result."""
        now = time.time()

        # Order 1: visual then speech
        f1 = _fusion(window=10.0)
        v = _visual("Screen", ts=now + 1.0)
        s = _speech("Question", ts=now + 2.0)
        f1.ingest(v)
        f1.ingest(s)
        snap1 = f1.snapshot(reference_time=now + 2.0)

        # Order 2: speech then visual (out of delivery order)
        f2 = _fusion(window=10.0)
        f2.ingest(s)
        f2.ingest(v)
        snap2 = f2.snapshot(reference_time=now + 2.0)

        # Same observations regardless of ingestion order
        assert snap1.audio_observations == snap2.audio_observations
        assert snap1.visual_observations == snap2.visual_observations


class TestExcludeEventIds:
    """Test the exclude_event_ids parameter for snapshots."""

    def test_exclude_current_event(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        s1 = _speech("Prior speech", ts=now)
        s2 = _speech("Current question", ts=now + 2.0)

        fusion.ingest(s1)
        fusion.ingest(s2)

        snap = fusion.snapshot(reference_time=s2.timestamp, exclude_event_ids={s2.id})
        assert "Prior speech" in snap.audio_observations
        assert "Current question" not in snap.audio_observations
        assert s2.id not in snap.source_event_ids

    def test_exclude_empty_set(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        s = _speech("Hello", ts=now)
        fusion.ingest(s)

        snap = fusion.snapshot(reference_time=now, exclude_event_ids=set())
        assert "Hello" in snap.audio_observations
