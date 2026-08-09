"""Multimodal Fusion Engine — temporal correlation of audio + visual events.

Maintains a bounded time window of recent observations. Produces immutable
snapshots that the ContextBuilder uses to assemble multimodal context.

Design constraints:
- Never invokes an LLM
- Never blocks audio or camera capture
- Never stores raw frames or audio bytes
- Bounded by time, event count, and character count
- Events are ordered by timestamp within the window
- Late events (outside the window) are rejected
- Duplicate visual descriptions reuse existing deduplication upstream
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from app.config import FusionConfig
from app.events.types import Event, Modality, SpeechEvent, VisualEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionObservation:
    """A single observation within the fusion window."""

    event_id: str
    timestamp: float
    modality: Modality
    content: str  # transcript or visual description
    confidence: float = 0.0


@dataclass(frozen=True)
class MultimodalSnapshot:
    """Immutable temporal snapshot of correlated multimodal observations."""

    timestamp: float
    window_start: float
    window_end: float
    audio_observations: tuple[str, ...] = ()
    visual_observations: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    total_characters: int = 0


class MultimodalFusion:
    """Bounded temporal fusion of audio and visual events.

    Thread-safe: ingest may be called from event handlers on the asyncio loop
    while snapshot() is called from the reasoning path.
    """

    def __init__(self, config: FusionConfig) -> None:
        self._enabled = config.enabled
        self._window_seconds = config.window_seconds
        self._max_events = config.max_events
        self._max_characters = config.max_characters

        self._observations: list[FusionObservation] = []
        self._lock = threading.Lock()

        self._counters = {
            "events_received": 0,
            "events_correlated": 0,
            "events_expired": 0,
            "events_dropped": 0,
            "snapshots_created": 0,
            "out_of_order_events": 0,
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def ingest(self, event: Event) -> None:
        """Ingest an event into the fusion window.

        Lightweight — no I/O, no model calls. Safe to call from event handlers.
        Rejects events outside the current time window (late arrivals).
        Enforces max_events bound by dropping oldest when full.
        """
        if not self._enabled:
            return

        if isinstance(event, SpeechEvent):
            content = event.transcript
            modality = Modality.AUDIO
        elif isinstance(event, VisualEvent):
            content = event.description
            modality = Modality.VISUAL
        else:
            return

        if not content or not content.strip():
            return

        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            self._counters["events_received"] += 1

            # Reject late events that fall outside the window
            if event.timestamp < window_start:
                self._counters["events_dropped"] += 1
                logger.debug(
                    "Fusion: dropped late event %s (%.1fs old, window=%.1fs)",
                    event.id[:12], now - event.timestamp, self._window_seconds,
                )
                return

            obs = FusionObservation(
                event_id=event.id,
                timestamp=event.timestamp,
                modality=modality,
                content=content.strip(),
                confidence=event.confidence,
            )

            # Detect out-of-order arrival
            if self._observations and event.timestamp < self._observations[-1].timestamp:
                self._counters["out_of_order_events"] += 1

            self._observations.append(obs)
            self._counters["events_correlated"] += 1

            # Expire old observations
            self._expire(now)

            # Enforce max_events bound
            while len(self._observations) > self._max_events:
                self._observations.pop(0)
                self._counters["events_dropped"] += 1

            # Enforce max_characters bound
            self._enforce_char_limit()

    def snapshot(
        self,
        reference_time: float | None = None,
        exclude_event_ids: set[str] | None = None,
    ) -> MultimodalSnapshot:
        """Produce a bounded snapshot of the current multimodal state.

        Orders observations by timestamp. Returns only observations within
        the active window relative to reference_time (default: now).

        Args:
            reference_time: Center of the window (default: wall-clock now).
                For reasoning about a SpeechEvent, pass event.timestamp so
                the snapshot represents the situation at the time the user spoke.
            exclude_event_ids: Event IDs to exclude from the snapshot (e.g. the
                current SpeechEvent whose transcript is already the user message).
        """
        now = reference_time if reference_time is not None else time.time()
        window_start = now - self._window_seconds
        exclude = exclude_event_ids or set()

        with self._lock:
            # Expire before snapshot
            self._expire(now)

            # Sort by timestamp (handles out-of-order arrivals)
            active = sorted(self._observations, key=lambda o: o.timestamp)

            # Filter to window and exclude specified events
            in_window = [
                o for o in active
                if o.timestamp >= window_start and o.event_id not in exclude
            ]

            audio = tuple(o.content for o in in_window if o.modality == Modality.AUDIO)
            visual = tuple(o.content for o in in_window if o.modality == Modality.VISUAL)
            event_ids = tuple(o.event_id for o in in_window)
            total_chars = sum(len(o.content) for o in in_window)

            self._counters["snapshots_created"] += 1

        return MultimodalSnapshot(
            timestamp=now,
            window_start=window_start,
            window_end=now,
            audio_observations=audio,
            visual_observations=visual,
            source_event_ids=event_ids,
            total_characters=total_chars,
        )

    def _expire(self, now: float) -> None:
        """Remove observations outside the time window. Caller holds lock."""
        window_start = now - self._window_seconds
        before = len(self._observations)
        self._observations = [
            o for o in self._observations if o.timestamp >= window_start
        ]
        expired = before - len(self._observations)
        if expired > 0:
            self._counters["events_expired"] += expired

    def _enforce_char_limit(self) -> None:
        """Drop oldest observations until total chars <= max. Caller holds lock."""
        total = sum(len(o.content) for o in self._observations)
        while total > self._max_characters and self._observations:
            removed = self._observations.pop(0)
            total -= len(removed.content)
            self._counters["events_dropped"] += 1
