"""Frame sampler — decides which captured frames reach the visual analyzer.

Responsibilities:
- Enforce a bounded frame queue (latest-frame policy: newest wins).
- Respect a configurable analysis interval.
- Discard frames older than max_frame_age_seconds.
- Expose metrics: captured, analyzed, dropped (stale), dropped (sampled).
"""
from __future__ import annotations

import logging
import time
from collections import deque

from app.perception.types import Frame

logger = logging.getLogger(__name__)


class FrameSampler:
    """Controls flow from camera capture rate to analysis rate.

    Bounded queue with latest-frame preference: when a new frame arrives
    and the queue is full, the oldest queued frame is evicted.

    analysis_interval_ms: minimum time between frames forwarded for analysis.
    max_frame_age_seconds: frames older than this are discarded as stale.
    max_queue_size: bounded queue capacity (1 = always latest frame only).
    """

    def __init__(
        self,
        *,
        analysis_interval_ms: float = 1000.0,
        max_frame_age_seconds: float = 2.0,
        max_queue_size: int = 2,
    ) -> None:
        self._interval_s = analysis_interval_ms / 1000.0
        self._max_age = max_frame_age_seconds
        self._queue: deque[Frame] = deque(maxlen=max(1, max_queue_size))
        self._last_analysis_time: float = 0.0

        self.frames_received = 0
        self.frames_sampled = 0
        self.frames_dropped_stale = 0
        self.frames_dropped_interval = 0
        self.frames_evicted = 0

    def submit(self, frame: Frame) -> None:
        """Accept a captured frame into the bounded queue."""
        self.frames_received += 1
        if len(self._queue) == self._queue.maxlen:
            self.frames_evicted += 1
        self._queue.append(frame)

    def next_frame(self) -> Frame | None:
        """Return the next frame eligible for analysis, or None.

        Enforces analysis interval and frame freshness.
        Discards stale frames until a fresh one is found or queue is empty.
        """
        now = time.time()

        if now - self._last_analysis_time < self._interval_s:
            return None

        while self._queue:
            frame = self._queue.popleft()
            if frame.age_seconds > self._max_age:
                self.frames_dropped_stale += 1
                continue

            self._last_analysis_time = now
            self.frames_sampled += 1
            return frame

        return None

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def metrics(self) -> dict[str, int]:
        return {
            "frames_received": self.frames_received,
            "frames_sampled": self.frames_sampled,
            "frames_dropped_stale": self.frames_dropped_stale,
            "frames_dropped_interval": self.frames_dropped_interval,
            "frames_evicted": self.frames_evicted,
        }
