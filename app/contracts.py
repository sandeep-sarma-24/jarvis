"""Shared pipeline contracts: request lifecycle, cancellation, latency marks.

Owned by the integrator. Subsystems import from here; they do not edit it.
See docs/architecture/06-engineering-contracts.md.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

# Latency stages, in pipeline order. All marks are wall-clock time.time() values.
STAGES = (
    "speech_end",        # capture time of the last speech chunk (t0 for every metric)
    "segment_emitted",   # endpointing decided the utterance is over
    "asr_done",
    "attention_done",
    "context_ready",     # retrieval + context assembly finished
    "llm_first_token",
    "first_sentence",    # first complete sentence handed to TTS
    "tts_first_chunk",   # first synthesized audio chunk produced
    "first_audio",       # first sample handed to the output device
    "llm_done",
    "playback_done",
)


@dataclass(frozen=True)
class Transcript:
    """One finished utterance, produced by the audio frontend."""

    text: str
    speech_start_time: float
    speech_end_time: float       # capture time of the last speech chunk (excludes trailing silence)
    segment_emitted_time: float  # when endpointing fired
    asr_latency_ms: float
    audio_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Turn:
    """One user request from end of speech to end of playback.

    turn_id is the SpeechEvent.correlation_id and is carried by every event,
    log line and audio chunk belonging to the request. `cancel` is the single
    cancellation signal: setting it must stop reasoning, synthesis and playback.
    """

    turn_id: str
    speech_end_time: float
    cancel: threading.Event = field(default_factory=threading.Event)
    marks: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.marks.setdefault("speech_end", self.speech_end_time)

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def mark(self, stage: str, t: float | None = None) -> None:
        """Record a stage timestamp. First write wins; safe from any thread."""
        if stage not in STAGES:
            raise ValueError(f"unknown latency stage: {stage!r}")
        self.marks.setdefault(stage, time.time() if t is None else t)

    def elapsed_ms(self, stage: str) -> float | None:
        """Milliseconds from end of speech to `stage`, or None if not reached."""
        t = self.marks.get(stage)
        return None if t is None else (t - self.speech_end_time) * 1000


def percentiles(values: list[float]) -> dict[str, float]:
    """Nearest-rank P50/P95/P99 plus n/min/max. Empty input gives n=0 only."""
    if not values:
        return {"n": 0}
    s = sorted(values)

    def rank(p: float) -> float:
        return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]

    return {"n": len(s), "min": s[0], "p50": rank(0.50), "p95": rank(0.95), "p99": rank(0.99), "max": s[-1]}


class LatencyRecorder:
    """Bounded in-memory record of finished turns, for logs and benchmarks."""

    def __init__(self, maxlen: int = 1000) -> None:
        self.turns: deque[dict[str, object]] = deque(maxlen=maxlen)

    def record(self, turn: Turn, outcome: str) -> dict[str, object]:
        """Store a finished turn. outcome: completed | cancelled | failed | no_response."""
        row: dict[str, object] = {"turn_id": turn.turn_id, "outcome": outcome}
        for stage in STAGES[1:]:
            ms = turn.elapsed_ms(stage)
            if ms is not None:
                row[stage] = round(ms, 1)
        self.turns.append(row)
        return row

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-stage percentiles of ms-since-speech-end across recorded turns."""
        return {
            stage: percentiles([r[stage] for r in self.turns if stage in r])  # type: ignore[misc]
            for stage in STAGES[1:]
        }
