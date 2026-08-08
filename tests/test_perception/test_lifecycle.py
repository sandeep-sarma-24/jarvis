"""Tests for perception pipeline lifecycle and failure isolation.

Ensures:
- Camera failure does not crash audio pipeline
- Pipeline starts and stops cleanly
- Camera unavailable → pipeline disabled gracefully
- End-to-end: FakeCamera → sampler → analyzer → VisualEvent → bus
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.events.bus import EventBus
from app.events.types import Event, Modality, SpeechEvent, VisualEvent
from app.perception.camera import CameraError, CameraState, FakeCamera
from app.perception.pipeline import PerceptionPipeline
from app.perception.sampler import FrameSampler
from app.perception.visual import ChangeDetector, EventDeduplicator, MockVisualAnalyzer


def _make_pipeline(
    bus: EventBus,
    *,
    fail_on_start: bool = False,
    fail_on_capture: bool = False,
    analysis_interval_ms: float = 0,
    capture_interval_ms: float = 10,
    dedup_window_seconds: float = 0.0,
    change_threshold: float = 0.0,
) -> PerceptionPipeline:
    camera = FakeCamera(
        width=4, height=4,
        fail_on_start=fail_on_start,
        fail_on_capture=fail_on_capture,
    )
    sampler = FrameSampler(
        analysis_interval_ms=analysis_interval_ms,
        max_frame_age_seconds=5.0,
        max_queue_size=2,
    )
    change_detector = ChangeDetector(change_threshold=change_threshold)
    analyzer = MockVisualAnalyzer()
    deduplicator = EventDeduplicator(dedup_window_seconds=dedup_window_seconds)

    return PerceptionPipeline(
        camera=camera,
        sampler=sampler,
        change_detector=change_detector,
        analyzer=analyzer,
        deduplicator=deduplicator,
        bus=bus,
        session_id="test-session",
        capture_interval_ms=capture_interval_ms,
    )


@pytest.mark.asyncio
async def test_pipeline_start_stop():
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus)
    await pipeline.start()
    await asyncio.sleep(0.1)  # let some frames flow
    await pipeline.stop()

    assert pipeline.frames_captured > 0
    assert pipeline.frames_analyzed > 0

    await bus.shutdown()


@pytest.mark.asyncio
async def test_pipeline_camera_unavailable():
    """Camera failure on start → pipeline disabled, no crash."""
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus, fail_on_start=True)
    await pipeline.start()  # should not raise

    assert pipeline.frames_captured == 0
    await pipeline.stop()
    await bus.shutdown()


@pytest.mark.asyncio
async def test_pipeline_capture_failure():
    """Camera capture returns None → pipeline continues without crashing."""
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus, fail_on_capture=True)
    await pipeline.start()
    await asyncio.sleep(0.1)
    await pipeline.stop()

    assert pipeline.frames_captured == 0
    assert pipeline.frames_analyzed == 0

    await bus.shutdown()


@pytest.mark.asyncio
async def test_camera_failure_does_not_kill_audio():
    """Camera fails → audio events still flow normally."""
    bus = EventBus(queue_size=16)

    speech_received = []

    async def speech_handler(event: Event) -> None:
        speech_received.append(event)

    bus.subscribe(SpeechEvent, speech_handler)
    await bus.start()

    # Camera fails on start
    pipeline = _make_pipeline(bus, fail_on_start=True)
    await pipeline.start()

    # Audio event still works
    se = SpeechEvent(
        source=Modality.AUDIO,
        transcript="Hello Jarvis",
        duration_seconds=1.0,
    )
    await bus.publish(se)
    await asyncio.sleep(0.05)

    assert len(speech_received) == 1
    assert speech_received[0].transcript == "Hello Jarvis"

    await pipeline.stop()
    await bus.shutdown()


@pytest.mark.asyncio
async def test_end_to_end_visual_event_flow():
    """FakeCamera → sampler → analyzer → VisualEvent → bus subscriber."""
    bus = EventBus(queue_size=64)

    visual_received: list[VisualEvent] = []

    async def visual_handler(event: Event) -> None:
        if isinstance(event, VisualEvent):
            visual_received.append(event)

    bus.subscribe(VisualEvent, visual_handler)
    await bus.start()

    # No dedup, no change threshold → every analyzed frame emits an event
    pipeline = _make_pipeline(
        bus,
        analysis_interval_ms=0,
        capture_interval_ms=10,
        dedup_window_seconds=0.0,
        change_threshold=0.0,
    )
    await pipeline.start()
    await asyncio.sleep(0.2)
    await pipeline.stop()

    assert len(visual_received) > 0

    for ve in visual_received:
        assert isinstance(ve, VisualEvent)
        assert ve.source == Modality.VISUAL
        assert ve.event_type in (
            "SCENE_CHANGE", "PERSON_DETECTED", "OBJECT_DETECTED",
            "TEXT_DETECTED", "VISUAL_SUMMARY",
        )
        assert ve.description
        assert ve.confidence > 0
        # No raw image bytes in event
        assert "data" not in ve.metadata
        assert isinstance(ve.metadata.get("analysis_latency_ms"), float)

    await bus.shutdown()


@pytest.mark.asyncio
async def test_deduplication_suppresses_events():
    """With dedup enabled, repeated identical events are suppressed."""
    bus = EventBus(queue_size=64)

    visual_received: list[VisualEvent] = []

    async def visual_handler(event: Event) -> None:
        if isinstance(event, VisualEvent):
            visual_received.append(event)

    bus.subscribe(VisualEvent, visual_handler)
    await bus.start()

    # Long dedup window → identical events suppressed
    pipeline = _make_pipeline(
        bus,
        analysis_interval_ms=0,
        capture_interval_ms=10,
        dedup_window_seconds=60.0,
        change_threshold=0.0,
    )
    await pipeline.start()
    await asyncio.sleep(0.3)
    await pipeline.stop()

    assert pipeline.visual_events_suppressed > 0
    assert pipeline.visual_events_emitted > 0

    await bus.shutdown()


@pytest.mark.asyncio
async def test_no_raw_frames_in_events():
    """Visual events must never contain raw frame bytes."""
    bus = EventBus(queue_size=64)

    visual_received: list[VisualEvent] = []

    async def visual_handler(event: Event) -> None:
        if isinstance(event, VisualEvent):
            visual_received.append(event)

    bus.subscribe(VisualEvent, visual_handler)
    await bus.start()

    pipeline = _make_pipeline(bus, dedup_window_seconds=0.0, change_threshold=0.0)
    await pipeline.start()
    await asyncio.sleep(0.15)
    await pipeline.stop()

    for ve in visual_received:
        # No raw bytes anywhere in the event
        assert isinstance(ve.description, str)
        assert "frame" not in str(type(ve.metadata.get("data", "")))
        for key, val in ve.metadata.items():
            assert not isinstance(val, (bytes, bytearray)), f"Raw bytes found in metadata[{key}]"

    await bus.shutdown()


@pytest.mark.asyncio
async def test_pipeline_metrics():
    bus = EventBus(queue_size=64)
    await bus.start()

    pipeline = _make_pipeline(bus, dedup_window_seconds=0.0, change_threshold=0.0)
    await pipeline.start()
    await asyncio.sleep(0.15)
    await pipeline.stop()

    m = pipeline.metrics
    assert m["frames_captured"] > 0
    assert m["frames_analyzed"] > 0
    assert m["visual_events_emitted"] > 0
    assert "frames_received" in m  # from sampler

    await bus.shutdown()
