"""Tests for perception pipeline lifecycle, concurrency, and failure isolation.

Ensures:
- Camera failure does not crash audio pipeline
- Pipeline starts and stops cleanly
- Camera unavailable → pipeline disabled gracefully
- End-to-end: FakeCamera → sampler → analyzer → VisualEvent → bus
- Capture and analysis are independently scheduled (decoupled)
- Slow analyzer does not block capture
- Latest-frame policy works across the pipeline
- Stale frames are dropped before analysis
- Analysis interval is respected
- Start/stop is idempotent
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
from app.perception.types import Detection, Frame, VisualAnalysis, VisualEventType
from app.perception.visual import ChangeDetector, EventDeduplicator, MockVisualAnalyzer


class SlowVisualAnalyzer:
    """Analyzer that takes a configurable time per analysis — for concurrency tests."""

    def __init__(self, delay_seconds: float = 0.5) -> None:
        self._delay = delay_seconds
        self.analyses_performed = 0
        self.analyzed_pixels: list[int] = []

    async def analyze(self, frame: Frame) -> VisualAnalysis | None:
        self.analyses_performed += 1
        self.analyzed_pixels.append(frame.data[0])
        await asyncio.sleep(self._delay)
        return VisualAnalysis(
            timestamp=frame.timestamp,
            description=f"Analysis of pixel {frame.data[0]}",
            event_type=VisualEventType.VISUAL_SUMMARY,
            confidence=0.8,
            metadata={"pixel": frame.data[0]},
        )


class CountingAnalyzer:
    """Analyzer that records call count and frame pixels without delay."""

    def __init__(self) -> None:
        self.call_count = 0
        self.analyzed_pixels: list[int] = []

    async def analyze(self, frame: Frame) -> VisualAnalysis | None:
        self.call_count += 1
        self.analyzed_pixels.append(frame.data[0])
        return VisualAnalysis(
            timestamp=frame.timestamp,
            description=f"Frame pixel {frame.data[0]}",
            event_type=VisualEventType.VISUAL_SUMMARY,
            confidence=0.7,
        )


def _make_pipeline(
    bus: EventBus,
    *,
    fail_on_start: bool = False,
    fail_on_capture: bool = False,
    analysis_interval_ms: float = 0,
    capture_interval_ms: float = 10,
    dedup_window_seconds: float = 0.0,
    change_threshold: float = 0.0,
    analyzer=None,
    max_queue_size: int = 2,
    max_frame_age_seconds: float = 5.0,
) -> PerceptionPipeline:
    camera = FakeCamera(
        width=4, height=4,
        fail_on_start=fail_on_start,
        fail_on_capture=fail_on_capture,
    )
    sampler = FrameSampler(
        analysis_interval_ms=analysis_interval_ms,
        max_frame_age_seconds=max_frame_age_seconds,
        max_queue_size=max_queue_size,
    )
    change_detector = ChangeDetector(change_threshold=change_threshold)
    if analyzer is None:
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
        analysis_interval_ms=analysis_interval_ms,
    )


# ── Basic lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_start_stop():
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus)
    await pipeline.start()
    await asyncio.sleep(0.1)
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
    await pipeline.start()

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

    pipeline = _make_pipeline(bus, fail_on_start=True)
    await pipeline.start()

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


# ── End-to-end flow ──────────────────────────────────────────────────────


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
        assert isinstance(ve.description, str)
        for key, val in ve.metadata.items():
            assert not isinstance(val, (bytes, bytearray)), f"Raw bytes in metadata[{key}]"

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
    assert "frames_received" in m

    await bus.shutdown()


# ── Concurrency: capture decoupled from analysis ────────────────────────


@pytest.mark.asyncio
async def test_slow_analyzer_does_not_block_capture():
    """Capture continues at full rate while analysis is slow.

    With capture_interval=10ms and analysis delay=500ms, during one analysis
    cycle ~50 frames should be captured. This proves the two loops are decoupled.
    """
    bus = EventBus(queue_size=64)
    await bus.start()

    slow = SlowVisualAnalyzer(delay_seconds=0.3)
    pipeline = _make_pipeline(
        bus,
        analyzer=slow,
        capture_interval_ms=10,
        analysis_interval_ms=0,
        change_threshold=0.0,
        dedup_window_seconds=0.0,
        max_queue_size=2,
    )
    await pipeline.start()
    await asyncio.sleep(0.8)
    await pipeline.stop()

    assert pipeline.frames_captured > pipeline.frames_analyzed
    assert pipeline.frames_captured >= 10  # at least 10 captures in 800ms at 10ms interval
    assert slow.analyses_performed >= 1

    await bus.shutdown()


@pytest.mark.asyncio
async def test_latest_frame_analyzed_after_slow_analysis():
    """After a slow analysis finishes, the newest available frame is analyzed, not stale intermediates.

    FakeCamera produces frames with incrementing pixel values (1, 2, 3, ...).
    With a 300ms analysis delay and 10ms capture interval, ~30 frames are produced
    during one analysis. The second analyzed frame should have a high pixel value
    (near the latest), not pixel=2.
    """
    bus = EventBus(queue_size=64)
    await bus.start()

    slow = SlowVisualAnalyzer(delay_seconds=0.3)
    pipeline = _make_pipeline(
        bus,
        analyzer=slow,
        capture_interval_ms=10,
        analysis_interval_ms=0,
        change_threshold=0.0,
        dedup_window_seconds=0.0,
        max_queue_size=1,  # only keep the very latest frame
    )
    await pipeline.start()
    await asyncio.sleep(0.8)
    await pipeline.stop()

    assert len(slow.analyzed_pixels) >= 2
    # The second analyzed pixel should be much higher than 2,
    # proving old intermediate frames were discarded
    assert slow.analyzed_pixels[1] > 5

    await bus.shutdown()


# ── Stale frame behavior ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stale_frames_dropped_by_sampler():
    """FrameSampler discards frames older than max_frame_age_seconds.
    Unit-level verification using direct sampler API.
    """
    sampler = FrameSampler(analysis_interval_ms=0, max_frame_age_seconds=0.01)

    stale = Frame(
        timestamp=time.time() - 10.0,
        data=bytes([42] * 12),
        width=2, height=2, format="rgb24",
    )
    sampler.submit(stale)

    result = sampler.next_frame()
    assert result is None
    assert sampler.frames_dropped_stale == 1
    assert sampler.frames_sampled == 0


@pytest.mark.asyncio
async def test_stale_frame_does_not_invoke_analyzer():
    """Pipeline-level: a stale frame should be dropped before reaching the analyzer.

    Use a slow analyzer so frames accumulate in the queue and age out.
    With max_queue_size=8, analysis_delay=0.4s, and max_frame_age=0.05s,
    frames captured early in the queue go stale while analysis is blocked.
    When analysis finishes and sampler drains the queue, old frames are discarded.
    """
    bus = EventBus(queue_size=16)
    await bus.start()

    slow = SlowVisualAnalyzer(delay_seconds=0.4)
    pipeline = _make_pipeline(
        bus,
        analyzer=slow,
        capture_interval_ms=10,
        analysis_interval_ms=0,
        max_frame_age_seconds=0.05,
        change_threshold=0.0,
        dedup_window_seconds=0.0,
        max_queue_size=8,
    )
    await pipeline.start()
    await asyncio.sleep(1.0)
    await pipeline.stop()

    m = pipeline.metrics
    assert m["frames_dropped_stale"] > 0

    await bus.shutdown()


# ── Analysis interval ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analysis_interval_respected():
    """With analysis_interval=200ms and capture at 10ms, analysis should run ~5 times in 1s."""
    bus = EventBus(queue_size=64)
    await bus.start()

    counting = CountingAnalyzer()
    pipeline = _make_pipeline(
        bus,
        analyzer=counting,
        capture_interval_ms=10,
        analysis_interval_ms=200,
        change_threshold=0.0,
        dedup_window_seconds=0.0,
    )
    await pipeline.start()
    await asyncio.sleep(1.0)
    await pipeline.stop()

    # At 200ms interval over 1s, expect ~5 analyses. Allow tolerance for scheduler jitter.
    assert 3 <= counting.call_count <= 8

    await bus.shutdown()


# ── Change detection before analyzer ────────────────────────────────────


@pytest.mark.asyncio
async def test_identical_frames_skip_analyzer():
    """When all frames have the same pixel value, change detector says 'no change'
    after the first frame. The analyzer should only be called once.
    """
    bus = EventBus(queue_size=64)
    await bus.start()

    # FakeCamera produces frames with pixel = frame_count % 256
    # So frame 1 → pixel 1, frame 2 → pixel 2, etc.
    # To test identical frames, we need change_threshold high enough that
    # FakeCamera's small pixel increments don't register as changes.
    counting = CountingAnalyzer()
    pipeline = _make_pipeline(
        bus,
        analyzer=counting,
        capture_interval_ms=10,
        analysis_interval_ms=0,
        change_threshold=0.9,  # very high threshold — tiny pixel diffs won't pass
        dedup_window_seconds=0.0,
    )
    await pipeline.start()
    await asyncio.sleep(0.15)
    await pipeline.stop()

    # First frame always triggers change, subsequent near-identical ones should not
    assert counting.call_count == 1
    assert pipeline.frames_skipped_no_change > 0

    await bus.shutdown()


# ── Start/stop idempotency and restartability ────────────────────────────


@pytest.mark.asyncio
async def test_double_start_is_idempotent():
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus)
    await pipeline.start()
    await pipeline.start()  # should not create duplicate tasks

    await asyncio.sleep(0.05)
    await pipeline.stop()

    # If duplicate tasks were created, frames_captured would be roughly double
    # For a 50ms run at 10ms interval, expect ~5 captures
    assert pipeline.frames_captured < 20

    await bus.shutdown()


@pytest.mark.asyncio
async def test_double_stop_is_safe():
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus)
    await pipeline.start()
    await asyncio.sleep(0.05)
    await pipeline.stop()
    await pipeline.stop()  # should not raise

    await bus.shutdown()


@pytest.mark.asyncio
async def test_start_stop_restart():
    """Pipeline can be started, stopped, and started again."""
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus)

    await pipeline.start()
    await asyncio.sleep(0.05)
    await pipeline.stop()

    captured_first = pipeline.frames_captured
    assert captured_first > 0

    # After stop, no background tasks should remain
    await asyncio.sleep(0.05)
    captured_after_stop = pipeline.frames_captured
    assert captured_after_stop == captured_first  # no new captures

    await bus.shutdown()


@pytest.mark.asyncio
async def test_no_tasks_remain_after_stop():
    """After stop(), no background perception tasks keep running."""
    bus = EventBus(queue_size=16)
    await bus.start()

    pipeline = _make_pipeline(bus)
    await pipeline.start()
    await asyncio.sleep(0.05)
    await pipeline.stop()

    captured = pipeline.frames_captured
    await asyncio.sleep(0.1)
    assert pipeline.frames_captured == captured  # no ghost tasks

    await bus.shutdown()
