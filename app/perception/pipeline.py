"""Perception pipeline — camera capture → sampling → analysis → visual events.

Wires together Camera, FrameSampler, ChangeDetector, VisualAnalyzer,
and EventDeduplicator into an async loop that publishes VisualEvents
to the EventBus.

Frames exist only as long as needed for analysis. No frame persistence,
no raw frames in logs, no cloud APIs.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from app.events.bus import EventBus
from app.events.types import Modality, VisualEvent
from app.perception.camera import Camera, CameraError, CameraState
from app.perception.sampler import FrameSampler
from app.perception.visual import ChangeDetector, EventDeduplicator, VisualAnalyzer

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """Async pipeline: Camera → FrameSampler → ChangeDetector → VisualAnalyzer → EventBus.

    Capture loop runs at capture_interval; analysis only happens when the sampler
    releases a frame AND the change detector confirms a meaningful difference.
    """

    def __init__(
        self,
        *,
        camera: Camera,
        sampler: FrameSampler,
        change_detector: ChangeDetector,
        analyzer: VisualAnalyzer,
        deduplicator: EventDeduplicator,
        bus: EventBus,
        session_id: str,
        capture_interval_ms: float = 100.0,
    ) -> None:
        self._camera = camera
        self._sampler = sampler
        self._change_detector = change_detector
        self._analyzer = analyzer
        self._deduplicator = deduplicator
        self._bus = bus
        self._session_id = session_id
        self._capture_interval_s = capture_interval_ms / 1000.0
        self._task: asyncio.Task | None = None

        self.frames_captured = 0
        self.frames_analyzed = 0
        self.frames_skipped_no_change = 0
        self.visual_events_emitted = 0
        self.visual_events_suppressed = 0
        self.capture_errors = 0
        self.analysis_errors = 0

    @property
    def metrics(self) -> dict[str, int | float]:
        return {
            "frames_captured": self.frames_captured,
            "frames_analyzed": self.frames_analyzed,
            "frames_skipped_no_change": self.frames_skipped_no_change,
            "visual_events_emitted": self.visual_events_emitted,
            "visual_events_suppressed": self.visual_events_suppressed,
            "capture_errors": self.capture_errors,
            "analysis_errors": self.analysis_errors,
            **self._sampler.metrics,
        }

    async def start(self) -> None:
        """Start the camera and begin the capture → analyze loop."""
        try:
            await self._camera.start()
        except CameraError:
            logger.warning("Camera unavailable — perception pipeline disabled")
            return

        self._task = asyncio.create_task(self._run())
        logger.info("Perception pipeline started")

    async def stop(self) -> None:
        """Stop the capture loop and shut down the camera."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._camera.state in (CameraState.RUNNING, CameraState.STARTING):
            try:
                await self._camera.stop()
            except Exception:
                logger.exception("Error stopping camera")

        logger.info("Perception pipeline stopped — %s", self.metrics)

    async def _run(self) -> None:
        try:
            while True:
                await self._tick()
                await asyncio.sleep(self._capture_interval_s)
        except asyncio.CancelledError:
            return

    async def _tick(self) -> None:
        """One capture-and-maybe-analyze cycle."""
        frame = await self._capture_frame()
        if frame is None:
            return

        self._sampler.submit(frame)

        candidate = self._sampler.next_frame()
        if candidate is None:
            return

        if not self._change_detector.has_changed(candidate):
            self.frames_skipped_no_change += 1
            return

        await self._analyze_and_emit(candidate)

    async def _capture_frame(self):
        try:
            frame = await self._camera.capture()
            if frame is not None:
                self.frames_captured += 1
            return frame
        except Exception:
            self.capture_errors += 1
            logger.exception("Frame capture error")
            return None

    async def _analyze_and_emit(self, frame) -> None:
        t0 = time.time()
        try:
            analysis = await self._analyzer.analyze(frame)
        except Exception:
            self.analysis_errors += 1
            logger.exception("Visual analysis error")
            return

        if analysis is None:
            return

        self.frames_analyzed += 1
        analysis_latency_ms = (time.time() - t0) * 1000

        event_type_name = analysis.event_type.name
        if self._deduplicator.is_duplicate(event_type_name, analysis.description):
            self.visual_events_suppressed += 1
            return

        event = VisualEvent(
            session_id=self._session_id,
            correlation_id=uuid.uuid4().hex,
            source=Modality.VISUAL,
            confidence=analysis.confidence,
            event_type=event_type_name,
            description=analysis.description,
            metadata={
                "analysis_latency_ms": round(analysis_latency_ms, 1),
                "frame_age_at_analysis_ms": round(frame.age_seconds * 1000, 1),
                "detections": [
                    {"label": d.label, "confidence": d.confidence}
                    for d in analysis.detections
                ],
            },
        )

        await self._bus.publish(event)
        self.visual_events_emitted += 1
        logger.debug(
            "VisualEvent emitted: type=%s desc=%r confidence=%.2f",
            event_type_name, analysis.description[:80], analysis.confidence,
        )
