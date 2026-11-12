"""TTS protocol and result types.

The rest of JARVIS depends only on this protocol, never on a specific TTS engine.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TTSResult:
    """Result of a TTS synthesis call."""

    success: bool
    audio: np.ndarray | None  # float32 waveform; None on failure
    sample_rate: int
    channels: int
    duration_seconds: float
    latency_ms: float
    model: str
    error: str | None = None


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of synthesized audio for streaming playback."""

    audio: np.ndarray  # float32 waveform
    sample_rate: int
    is_last: bool = False


@runtime_checkable
class TextToSpeech(Protocol):
    """Abstract interface for TTS engines.

    Implementations may be synchronous — the caller is responsible for
    running synthesis off the event loop via asyncio.to_thread.
    """

    @property
    def model_name(self) -> str: ...

    def synthesize(self, text: str) -> TTSResult: ...

    def synthesize_stream(self, text: str) -> Iterator[AudioChunk]: ...

    def close(self) -> None: ...


@runtime_checkable
class AudioPlayer(Protocol):
    """Abstract interface for audio playback.

    Must support non-blocking play and immediate stop (interruption).
    Exposes on_done callback for completion/interruption reporting.
    """

    def play(self, audio: np.ndarray, sample_rate: int) -> None: ...

    def stop(self) -> None: ...

    def is_playing(self) -> bool: ...

    def close(self) -> None: ...


class TTSError(RuntimeError):
    """Raised by streaming synthesis on failure."""


@dataclass(frozen=True)
class TTSStreamStats:
    first_chunk_ms: float | None  # None if no chunk was produced
    synth_ms: float
    audio_seconds: float
    chunks: int
    cancelled: bool


@runtime_checkable
class StreamingTextToSpeech(TextToSpeech, Protocol):
    """Incremental synthesis pushed into a sink (normally the audio player).

    stream_to() runs synthesis on the engine's own thread and calls
    sink(chunk) from that thread for every chunk, in order. It stops when the
    sink returns False or `cancel` is set, and raises TTSError on failure.
    AudioChunk.audio is 1-D mono float32 in [-1, 1] at `sample_rate`.
    """

    @property
    def sample_rate(self) -> int: ...

    def load(self) -> None:
        """Blocking model load. Call off the event loop."""
        ...

    async def stream_to(
        self,
        text: str,
        sink: Callable[[AudioChunk], bool],
        *,
        cancel: threading.Event | None = None,
    ) -> TTSStreamStats: ...


@runtime_checkable
class StreamingAudioPlayer(Protocol):
    """One persistent output stream fed by a bounded queue of turn-tagged chunks.

    Thread-safe. Callbacks (constructor args on_first_audio(turn_id) and
    on_done(turn_id, outcome)) fire on a non-event-loop thread and must be cheap.
    Audio whose turn_id is not the current turn is never played.
    """

    def start(self, sample_rate: int) -> None:
        """Open the output device once. Blocking; call off the event loop."""
        ...

    def begin(self, turn_id: str) -> None:
        """Make turn_id current, discarding any audio from other turns."""
        ...

    def feed(self, chunk: AudioChunk, turn_id: str) -> bool:
        """Queue a chunk. Blocks while the bounded queue is full (backpressure).

        Returns False without queueing if turn_id is not current; stop() wakes
        blocked feeders, which then return False.
        """
        ...

    def end(self, turn_id: str) -> None:
        """No more chunks for this turn; on_done(COMPLETED) fires once drained."""
        ...

    def stop(self) -> None:
        """Non-blocking. Drop queued audio and silence output within one device block."""
        ...

    def is_playing(self) -> bool: ...

    @property
    def stats(self) -> dict[str, float]:
        """queue_seconds, max_queue_seconds, chunks_dropped_stale, underruns."""
        ...

    def close(self) -> None: ...
