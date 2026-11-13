"""Visual analyzer abstraction, mock implementation, change detection, and deduplication.

VisualAnalyzer protocol: pluggable analysis backend.
MockVisualAnalyzer: deterministic results for testing and pipeline validation.
ChangeDetector: cheap frame-to-frame difference check (resized pixel diff).
EventDeduplicator: suppresses repeated identical visual events within a window.

Future: LocalVisionAnalyzer backed by a Gemma/Gemini-compatible local model
plugs in behind the same VisualAnalyzer protocol.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Protocol, runtime_checkable

from app.perception.types import (
    Detection,
    Frame,
    VisualAnalysis,
    VisualEventType,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class VisualAnalyzer(Protocol):
    async def analyze(self, frame: Frame, hint: str = "") -> VisualAnalysis | None: ...


class MockVisualAnalyzer:
    """Deterministic analyzer for testing — no model, no GPU.

    Returns canned results based on frame pixel content.
    """

    def __init__(self) -> None:
        self.analyses_performed = 0

    async def analyze(self, frame: Frame, hint: str = "") -> VisualAnalysis | None:
        self.analyses_performed += 1

        if not frame.data:
            return None

        pixel_val = frame.data[0]

        if pixel_val % 5 == 0:
            return VisualAnalysis(
                timestamp=frame.timestamp,
                description="Person detected near desk",
                event_type=VisualEventType.PERSON_DETECTED,
                detections=(Detection(label="person", confidence=0.92),),
                confidence=0.92,
                metadata={"mock": True, "frame_pixel": pixel_val},
            )
        elif pixel_val % 3 == 0:
            return VisualAnalysis(
                timestamp=frame.timestamp,
                description="Scene changed significantly",
                event_type=VisualEventType.SCENE_CHANGE,
                confidence=0.85,
                metadata={"mock": True, "frame_pixel": pixel_val},
            )
        else:
            return VisualAnalysis(
                timestamp=frame.timestamp,
                description="Desk with monitor and keyboard",
                event_type=VisualEventType.VISUAL_SUMMARY,
                confidence=0.70,
                metadata={"mock": True, "frame_pixel": pixel_val},
            )


class ChangeDetector:
    """Cheap frame-to-frame change detection via resized pixel difference.

    Compares a low-resolution grayscale summary of consecutive frames.
    If the mean absolute difference exceeds the threshold, the scene has changed.

    ponytail: simple mean-pixel-diff, not perceptual hashing or SSIM.
    Upgrade path: perceptual hash or structural similarity if needed.
    """

    def __init__(self, *, change_threshold: float = 0.05, summary_size: int = 64) -> None:
        self._threshold = change_threshold
        self._summary_size = summary_size
        self._last_summary: bytes | None = None
        self.checks_performed = 0
        self.changes_detected = 0

    def has_changed(self, frame: Frame) -> bool:
        """Return True if the frame differs meaningfully from the last checked frame."""
        self.checks_performed += 1
        summary = self._summarize(frame)

        if self._last_summary is None:
            self._last_summary = summary
            self.changes_detected += 1
            return True

        diff = self._pixel_diff(self._last_summary, summary)
        self._last_summary = summary

        changed = diff >= self._threshold
        if changed:
            self.changes_detected += 1
        return changed

    def _summarize(self, frame: Frame) -> bytes:
        """Downsample frame to a small grayscale summary for fast comparison."""
        n = min(len(frame.data), self._summary_size)
        if n == 0:
            return b"\x00" * self._summary_size

        step = max(1, len(frame.data) // self._summary_size)
        return bytes(frame.data[i] for i in range(0, min(len(frame.data), step * self._summary_size), step))

    def _pixel_diff(self, a: bytes, b: bytes) -> float:
        """Mean absolute difference normalized to [0, 1]."""
        length = min(len(a), len(b))
        if length == 0:
            return 1.0
        total = sum(abs(a[i] - b[i]) for i in range(length))
        return total / (length * 255.0)


class EventDeduplicator:
    """Suppresses repeated identical visual events within a time window.

    Events are fingerprinted by (event_type, description). If the same
    fingerprint appears within dedup_window_seconds, the event is suppressed.
    """

    def __init__(self, *, dedup_window_seconds: float = 30.0) -> None:
        self._window = dedup_window_seconds
        self._seen: dict[str, float] = {}
        self.events_checked = 0
        self.events_suppressed = 0

    def is_duplicate(self, event_type: str, description: str) -> bool:
        self.events_checked += 1
        fingerprint = self._fingerprint(event_type, description)
        now = time.time()

        self._prune(now)

        if fingerprint in self._seen:
            self.events_suppressed += 1
            return True

        self._seen[fingerprint] = now
        return False

    def _fingerprint(self, event_type: str, description: str) -> str:
        raw = f"{event_type}:{description}"
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    def _prune(self, now: float) -> None:
        expired = [k for k, t in self._seen.items() if now - t > self._window]
        for k in expired:
            del self._seen[k]
