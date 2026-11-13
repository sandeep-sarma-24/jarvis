"""Ambient vision pauses during voice turns; camera reads never overlap."""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from app.events.bus import EventBus
from app.perception.camera import CameraState, OpenCVCamera
from app.perception.pipeline import PerceptionPipeline
from app.perception.sampler import FrameSampler
from app.perception.visual import ChangeDetector, EventDeduplicator

from tests.test_perception.test_lifecycle import CountingAnalyzer, FakeCamera


async def test_ambient_analysis_waits_while_turn_active():
    idle = asyncio.Event()  # cleared: a voice turn is in flight
    analyzer = CountingAnalyzer()
    pipeline = PerceptionPipeline(
        camera=FakeCamera(width=4, height=4),
        sampler=FrameSampler(analysis_interval_ms=0, max_frame_age_seconds=5.0, max_queue_size=2),
        change_detector=ChangeDetector(change_threshold=0.0),
        analyzer=analyzer,
        deduplicator=EventDeduplicator(dedup_window_seconds=0.0),
        bus=EventBus(), session_id="s",
        capture_interval_ms=10, analysis_interval_ms=0, idle=idle,
    )
    await pipeline.start()
    await asyncio.sleep(0.1)
    assert pipeline.frames_captured > 0      # capture keeps running
    assert analyzer.call_count == 0
    idle.set()
    await asyncio.sleep(0.1)
    assert analyzer.call_count > 0
    await pipeline.stop()


class _OverlapCap:
    def __init__(self):
        self.active = 0
        self.overlaps = 0
        self.reads = 0
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            self.active += 1
            if self.active > 1:
                self.overlaps += 1
        time.sleep(0.01)
        with self._lock:
            self.active -= 1
            self.reads += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        pass


async def test_camera_reads_are_serialised():
    cam = OpenCVCamera()
    cap = _OverlapCap()
    cam._cap = cap
    cam._state = CameraState.RUNNING
    frames = await asyncio.gather(*(cam.capture() for _ in range(6)))
    assert all(f is not None for f in frames)
    assert cap.reads == 6 and cap.overlaps == 0
