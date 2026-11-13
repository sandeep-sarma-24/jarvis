"""Perception pipeline — decoupled camera capture and visual analysis.

Architecture:
    capture task → FrameSampler (bounded/latest-frame buffer) → analysis task

The capture loop and analysis loop run as independent asyncio tasks.
A slow visual analyzer never blocks camera capture. The sampler enforces
freshness, analysis interval, and bounded queue semantics between them.

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
    """Decoupled capture + analysis pipeline.

    Two independent async tasks:
      _capture_loop: captures frames at capture_interval, submits to FrameSampler.
      _analysis_loop: polls sampler for eligible frames, runs change detection,
                      visual analysis, deduplication, and publishes VisualEvents.

    A slow analyzer does not block capture. The sampler's bounded queue
    ensures the analysis loop always gets the freshest available frame.
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
        analysis_interval_ms: float = 1000.0,
        idle: asyncio.Event | None = None,
        scene=None,
    ) -> None:
        self._camera = camera
        self._sampler = sampler
        self._change_detector = change_detector
        self._analyzer = analyzer
        self._deduplicator = deduplicator
        self._bus = bus
        self._session_id = session_id
        self._capture_interval_s = capture_interval_ms / 1000.0
        self._analysis_interval_s = analysis_interval_ms / 1000.0
        # When given, ambient analysis only starts while this is set (no voice
        # turn in flight), so it never competes with reasoning for the GPU.
        self._idle = idle
        # A loaded SceneWatcher replaces the capture/analysis loops: the detector gates the VLM.
        self._scene = scene
        if scene is not None:
            scene.describe = self._describe
        self._capture_task: asyncio.Task | None = None
        self._analysis_task: asyncio.Task | None = None

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
            **(self._scene.metrics if self._scene is not None else {}),
        }

    async def start(self) -> None:
        """Start the camera, then spawn capture and analysis tasks."""
        if self._capture_task is not None:
            return

        try:
            await self._camera.start()
        except CameraError:
            logger.warning("Camera unavailable — perception pipeline disabled")
            return

        if self._scene is not None and self._scene.available:
            self._analysis_task = asyncio.create_task(self._scene.run())
            logger.info("Perception pipeline started (detector-gated VLM)")
            return
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("Perception pipeline started (capture + analysis)")

    async def stop(self) -> None:
        """Cancel both tasks, stop camera, release resources."""
        tasks = [t for t in (self._capture_task, self._analysis_task) if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._capture_task = None
        self._analysis_task = None

        if self._camera.state in (CameraState.RUNNING, CameraState.STARTING):
            try:
                await self._camera.stop()
            except Exception:
                logger.exception("Error stopping camera")

        logger.info("Perception pipeline stopped — %s", self.metrics)

    # ── Capture loop ─────────────────────────────────────────────────────

    async def _capture_loop(self) -> None:
        """Capture frames at capture_interval, submit to sampler. Never blocks on analysis."""
        try:
            while True:
                try:
                    frame = await self._camera.capture()
                    if frame is not None:
                        self.frames_captured += 1
                        self._sampler.submit(frame)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.capture_errors += 1
                    logger.exception("Frame capture error")

                await asyncio.sleep(self._capture_interval_s)
        except asyncio.CancelledError:
            return

    # ── Analysis loop ────────────────────────────────────────────────────

    async def _analysis_loop(self) -> None:
        """Poll sampler for eligible frames, analyze, publish events.

        The sampler enforces analysis interval and freshness. This loop
        polls at a rate slightly faster than the analysis interval to stay
        responsive, but the sampler gates actual analysis.
        """
        poll_interval = min(self._analysis_interval_s, self._capture_interval_s * 2)
        poll_interval = max(poll_interval, 0.01)

        try:
            while True:
                if self._idle is not None:
                    await self._idle.wait()
                candidate = self._sampler.next_frame()
                if candidate is None:
                    await asyncio.sleep(poll_interval)
                    continue

                if not self._change_detector.has_changed(candidate):
                    self.frames_skipped_no_change += 1
                    await asyncio.sleep(poll_interval)
                    continue

                await self._analyze_and_emit(candidate)
        except asyncio.CancelledError:
            return

    async def _describe(self, frame, hint: str) -> None:
        await self._analyze_and_emit(frame, hint)

    async def _analyze_and_emit(self, frame, hint: str = "") -> None:
        t0 = time.time()
        try:
            analysis = await (self._analyzer.analyze(frame, hint=hint) if hint else self._analyzer.analyze(frame))
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
        logger.info(
            "VISUAL type=%s conf=%.2f desc=%r",
            event_type_name, analysis.confidence, analysis.description,
        )
