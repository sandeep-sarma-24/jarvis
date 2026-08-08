"""TTS protocol and result types.

The rest of JARVIS depends only on this protocol, never on a specific TTS engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

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
    """

    def play(self, audio: np.ndarray, sample_rate: int) -> None: ...

    def stop(self) -> None: ...

    def is_playing(self) -> bool: ...

    def close(self) -> None: ...
