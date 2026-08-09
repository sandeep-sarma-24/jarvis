"""Tests for multimodal fusion engine — Phase 8."""

from __future__ import annotations

import time

import pytest

from app.config import FusionConfig
from app.events.types import Modality, SpeechEvent, VisualEvent
from app.fusion.engine import MultimodalFusion, MultimodalSnapshot


def _speech(transcript: str, ts: float = 0.0, **kw) -> SpeechEvent:
    return SpeechEvent(
        timestamp=ts or time.time(),
        source=Modality.AUDIO,
        transcript=transcript,
        **kw,
    )


def _visual(description: str, ts: float = 0.0, event_type: str = "SCENE_CHANGE", **kw) -> VisualEvent:
    return VisualEvent(
        timestamp=ts or time.time(),
        source=Modality.VISUAL,
        description=description,
        event_type=event_type,
        confidence=0.9,
        **kw,
    )


def _fusion(window: float = 10.0, max_events: int = 20, max_chars: int = 3000) -> MultimodalFusion:
    return MultimodalFusion(FusionConfig(
        enabled=True, window_seconds=window, max_events=max_events, max_characters=max_chars,
    ))


class TestTemporalCorrelation:
    """Speech + visual correlation within the time window."""

    def test_speech_and_visual_correlated_within_window(self):
        """Scenario from spec: visual at t=100, speech at t=103, window=10s."""
        now = time.time()
        fusion = _fusion(window=10.0)

        visual = _visual("Terminal window with Python traceback.", ts=now)
        speech = _speech("What is wrong with this?", ts=now + 3.0)

        fusion.ingest(visual)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=now + 3.0)
        assert "Terminal window with Python traceback." in snap.visual_observations
        assert "What is wrong with this?" in snap.audio_observations
        assert visual.id in snap.source_event_ids
        assert speech.id in snap.source_event_ids

    def test_visual_not_correlated_outside_window(self):
        """Scenario from spec: visual at t=100, speech at t=130, window=10s."""
        now = time.time()
        fusion = _fusion(window=10.0)

        visual = _visual("Terminal window with Python traceback.", ts=now)
        fusion.ingest(visual)

        # 30 seconds later
        speech = _speech("What is wrong with this?", ts=now + 30.0)
        fusion.ingest(speech)

        snap = fusion.snapshot(reference_time=now + 30.0)
        assert snap.visual_observations == ()
        assert "What is wrong with this?" in snap.audio_observations

    def test_visual_then_speech_correlation(self):
        now = time.time()
        fusion = _fusion(window=15.0)

        fusion.ingest(_visual("Coffee mug on desk.", ts=now))
        fusion.ingest(_speech("What is that?", ts=now + 2.0))

        snap = fusion.snapshot(reference_time=now + 2.0)
        assert "Coffee mug on desk." in snap.visual_observations
        assert "What is that?" in snap.audio_observations

    def test_speech_then_visual_correlation(self):
        now = time.time()
        fusion = _fusion(window=15.0)

        fusion.ingest(_speech("Look at this.", ts=now))
        fusion.ingest(_visual("Python traceback visible on screen.", ts=now + 1.0))

        snap = fusion.snapshot(reference_time=now + 1.0)
        assert "Look at this." in snap.audio_observations
        assert "Python traceback visible on screen." in snap.visual_observations


class TestTimeWindow:
    """Events inside and outside the time window."""

    def test_events_inside_window_included(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("Hello", ts=now - 5.0))
        fusion.ingest(_visual("Person at desk", ts=now - 3.0))

        snap = fusion.snapshot(reference_time=now)
        assert "Hello" in snap.audio_observations
        assert "Person at desk" in snap.visual_observations

    def test_events_outside_window_excluded(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("Old speech", ts=now - 20.0))
        fusion.ingest(_visual("Old visual", ts=now - 15.0))
        fusion.ingest(_speech("Recent speech", ts=now - 2.0))

        snap = fusion.snapshot(reference_time=now)
        assert "Old speech" not in snap.audio_observations
        assert "Old visual" not in snap.visual_observations
        assert "Recent speech" in snap.audio_observations


class TestExpiration:
    """Event expiration when window moves."""

    def test_events_expire_on_ingest(self):
        now = time.time()
        fusion = _fusion(window=5.0)

        fusion.ingest(_speech("Early", ts=now))

        # After window passes, next ingest expires old
        late = _speech("Late", ts=now + 10.0)
        fusion.ingest(late)

        snap = fusion.snapshot(reference_time=now + 10.0)
        assert "Early" not in snap.audio_observations
        assert "Late" in snap.audio_observations
        assert fusion.counters["events_expired"] >= 1

    def test_events_expire_on_snapshot(self):
        now = time.time()
        fusion = _fusion(window=5.0)

        fusion.ingest(_speech("Will expire", ts=now))

        snap = fusion.snapshot(reference_time=now + 10.0)
        assert snap.audio_observations == ()


class TestBounds:
    """Max event and max character bounds."""

    def test_max_events_enforced(self):
        fusion = _fusion(window=60.0, max_events=3)
        now = time.time()

        for i in range(5):
            fusion.ingest(_speech(f"msg{i}", ts=now + i))

        snap = fusion.snapshot(reference_time=now + 5.0)
        total = len(snap.audio_observations) + len(snap.visual_observations)
        assert total <= 3

    def test_max_characters_enforced(self):
        fusion = _fusion(window=60.0, max_chars=50)
        now = time.time()

        fusion.ingest(_speech("A" * 30, ts=now))
        fusion.ingest(_speech("B" * 30, ts=now + 1.0))

        snap = fusion.snapshot(reference_time=now + 1.0)
        assert snap.total_characters <= 50

    def test_oldest_dropped_on_overflow(self):
        fusion = _fusion(window=60.0, max_events=2)
        now = time.time()

        fusion.ingest(_speech("first", ts=now))
        fusion.ingest(_speech("second", ts=now + 1.0))
        fusion.ingest(_speech("third", ts=now + 2.0))

        snap = fusion.snapshot(reference_time=now + 2.0)
        assert "first" not in snap.audio_observations
        assert "third" in snap.audio_observations


class TestTimestampOrdering:
    """Timestamp ordering and out-of-order events."""

    def test_observations_ordered_by_timestamp(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("Second", ts=now + 2.0))
        fusion.ingest(_speech("First", ts=now + 1.0))
        fusion.ingest(_speech("Third", ts=now + 3.0))

        snap = fusion.snapshot(reference_time=now + 3.0)
        assert snap.audio_observations == ("First", "Second", "Third")

    def test_out_of_order_counted(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("Later", ts=now + 5.0))
        fusion.ingest(_speech("Earlier", ts=now + 2.0))

        assert fusion.counters["out_of_order_events"] == 1

    def test_mixed_modality_ordering(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_visual("Visual first", ts=now + 1.0))
        fusion.ingest(_speech("Speech second", ts=now + 2.0))
        fusion.ingest(_visual("Visual third", ts=now + 3.0))

        snap = fusion.snapshot(reference_time=now + 3.0)
        assert snap.visual_observations == ("Visual first", "Visual third")
        assert snap.audio_observations == ("Speech second",)


class TestLateEvents:
    """Late events are rejected from current context."""

    def test_late_event_rejected(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        # Ingest a current event first to establish the window
        fusion.ingest(_speech("Current", ts=now))

        # Try to ingest a very old event
        late = _visual("Very old", ts=now - 30.0)
        fusion.ingest(late)

        snap = fusion.snapshot(reference_time=now)
        assert "Very old" not in snap.visual_observations
        assert fusion.counters["events_dropped"] >= 1

    def test_barely_in_window_accepted(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        # Event at window boundary
        fusion.ingest(_speech("Edge case", ts=now - 9.5))

        snap = fusion.snapshot(reference_time=now)
        assert "Edge case" in snap.audio_observations


class TestSourceEventIDs:
    """Source event ID tracking."""

    def test_source_ids_tracked(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        s = _speech("Hello", ts=now)
        v = _visual("World", ts=now + 1.0)

        fusion.ingest(s)
        fusion.ingest(v)

        snap = fusion.snapshot(reference_time=now + 1.0)
        assert s.id in snap.source_event_ids
        assert v.id in snap.source_event_ids


class TestImmutability:
    """Events are not mutated by fusion."""

    def test_speech_event_not_mutated(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        event = _speech("Test", ts=now)
        original_id = event.id
        original_ts = event.timestamp
        original_transcript = event.transcript

        fusion.ingest(event)

        assert event.id == original_id
        assert event.timestamp == original_ts
        assert event.transcript == original_transcript

    def test_visual_event_not_mutated(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        event = _visual("Scene", ts=now)
        original_id = event.id
        original_ts = event.timestamp
        original_desc = event.description

        fusion.ingest(event)

        assert event.id == original_id
        assert event.timestamp == original_ts
        assert event.description == original_desc

    def test_snapshot_is_frozen(self):
        now = time.time()
        fusion = _fusion(window=10.0)
        fusion.ingest(_speech("Test", ts=now))

        snap = fusion.snapshot(reference_time=now)
        with pytest.raises((TypeError, AttributeError)):
            snap.timestamp = 999.0  # type: ignore[misc]


class TestEmptyState:
    """Empty fusion state produces empty snapshot."""

    def test_empty_fusion_snapshot(self):
        fusion = _fusion(window=10.0)
        snap = fusion.snapshot()

        assert snap.audio_observations == ()
        assert snap.visual_observations == ()
        assert snap.source_event_ids == ()
        assert snap.total_characters == 0


class TestDisabled:
    """Fusion disabled mode."""

    def test_disabled_does_not_ingest(self):
        fusion = MultimodalFusion(FusionConfig(enabled=False))
        fusion.ingest(_speech("Should not be stored"))

        snap = fusion.snapshot()
        assert snap.audio_observations == ()
        assert fusion.counters["events_received"] == 0


class TestNoLLMInvocation:
    """Fusion never invokes an LLM — verified by absence of model calls."""

    def test_ingest_is_pure_data_operation(self):
        """Fusion.ingest only manipulates in-memory lists, no external calls."""
        now = time.time()
        fusion = _fusion(window=10.0)

        # Ingest many events — should complete in microseconds, not seconds
        t0 = time.time()
        for i in range(100):
            fusion.ingest(_speech(f"msg{i}", ts=now + i * 0.1))
            fusion.ingest(_visual(f"vis{i}", ts=now + i * 0.1 + 0.05))
        elapsed_ms = (time.time() - t0) * 1000

        # Must be < 100ms for 200 events (no network/model calls)
        assert elapsed_ms < 100.0

    def test_snapshot_is_pure_data_operation(self):
        now = time.time()
        fusion = _fusion(window=10.0)
        for i in range(20):
            fusion.ingest(_speech(f"msg{i}", ts=now + i * 0.1))

        t0 = time.time()
        for _ in range(100):
            fusion.snapshot(reference_time=now + 2.0)
        elapsed_ms = (time.time() - t0) * 1000

        assert elapsed_ms < 100.0


class TestNoRawData:
    """Fusion never stores raw frames or audio bytes."""

    def test_no_frame_data_in_snapshot(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        # VisualEvent only carries description text, not raw frames
        v = _visual("Person at desk", ts=now)
        fusion.ingest(v)

        snap = fusion.snapshot(reference_time=now)
        # Snapshot only contains string descriptions
        for obs in snap.visual_observations:
            assert isinstance(obs, str)

    def test_no_audio_bytes_in_snapshot(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        s = _speech("Hello world", ts=now)
        fusion.ingest(s)

        snap = fusion.snapshot(reference_time=now)
        for obs in snap.audio_observations:
            assert isinstance(obs, str)


class TestDuplicateVisual:
    """Duplicate visual descriptions — dedup happens upstream, fusion stores what it receives."""

    def test_duplicate_visuals_stored_if_received(self):
        """Fusion trusts upstream deduplication; it stores whatever it receives."""
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_visual("Person at desk", ts=now))
        fusion.ingest(_visual("Person at desk", ts=now + 1.0))

        snap = fusion.snapshot(reference_time=now + 1.0)
        # Both are stored — upstream EventDeduplicator prevents this in practice
        assert len(snap.visual_observations) == 2


class TestMetrics:
    """Fusion metrics counters."""

    def test_events_received_counted(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("A", ts=now))
        fusion.ingest(_visual("B", ts=now))

        assert fusion.counters["events_received"] == 2

    def test_correlated_counted(self):
        now = time.time()
        fusion = _fusion(window=10.0)

        fusion.ingest(_speech("A", ts=now))
        assert fusion.counters["events_correlated"] == 1

    def test_expired_counted(self):
        now = time.time()
        fusion = _fusion(window=5.0)

        # Ingest an event that's currently within the window
        fusion.ingest(_speech("Old", ts=now - 3.0))
        assert fusion.counters["events_correlated"] == 1

        # Verify it expires when snapshot is called after the window passes
        snap = fusion.snapshot(reference_time=now + 5.0)
        assert fusion.counters["events_expired"] >= 1
        assert snap.audio_observations == ()

    def test_dropped_counted_on_max_events(self):
        fusion = _fusion(window=60.0, max_events=2)
        now = time.time()

        fusion.ingest(_speech("A", ts=now))
        fusion.ingest(_speech("B", ts=now + 1.0))
        fusion.ingest(_speech("C", ts=now + 2.0))

        assert fusion.counters["events_dropped"] >= 1

    def test_snapshots_created_counted(self):
        now = time.time()
        fusion = _fusion(window=10.0)
        fusion.ingest(_speech("X", ts=now))

        fusion.snapshot(reference_time=now)
        fusion.snapshot(reference_time=now)

        assert fusion.counters["snapshots_created"] == 2

    def test_empty_content_not_counted(self):
        fusion = _fusion(window=10.0)
        fusion.ingest(_speech("", ts=time.time()))
        fusion.ingest(_speech("   ", ts=time.time()))

        assert fusion.counters["events_received"] == 0


class TestSnapshotWindow:
    """Snapshot window_start and window_end fields."""

    def test_snapshot_window_bounds(self):
        now = time.time()
        fusion = _fusion(window=10.0)
        fusion.ingest(_speech("X", ts=now))

        snap = fusion.snapshot(reference_time=now)
        assert snap.window_end == now
        assert snap.window_start == pytest.approx(now - 10.0)
