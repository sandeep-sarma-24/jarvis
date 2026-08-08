"""Speech segmentation — state machine that groups VAD results into utterances.

Responsibility: "Does this sequence of VAD results form one utterance?"
Consumes VADResult stream, emits SpeechSegment when an utterance is complete.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from app.config import VADConfig
from app.audio.vad import VADResult


class _State(Enum):
    SILENCE = auto()
    SPEECH = auto()
    TRAILING_SILENCE = auto()


@dataclass
class SpeechSegment:
    """A completed utterance ready for ASR."""

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    start_time: float
    end_time: float


class SpeechSegmenter:
    """State machine: SILENCE → SPEECH → TRAILING_SILENCE → emit segment."""

    def __init__(self, config: VADConfig, sample_rate: int = 16000, chunk_samples: int = 512) -> None:
        self._cfg = config
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._chunk_duration_ms = (chunk_samples / sample_rate) * 1000

        self._speech_start_chunks = max(1, int(config.minimum_speech_ms / self._chunk_duration_ms / 3))
        self._silence_end_chunks = max(1, int(config.speech_end_silence_ms / self._chunk_duration_ms))
        self._max_chunks = int(config.maximum_segment_seconds * sample_rate / chunk_samples)

        self._state = _State.SILENCE
        self._speech_chunks: list[np.ndarray] = []
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._segment_start: float = 0.0

        self.segments_detected = 0

    def process(self, chunk: np.ndarray, vad_result: VADResult) -> SpeechSegment | None:
        """Feed one chunk + its VAD result. Returns a SpeechSegment when complete, else None."""
        now = time.time()

        if self._state == _State.SILENCE:
            if vad_result.is_speech:
                self._consecutive_speech += 1
                self._speech_chunks.append(chunk.copy())
                if self._consecutive_speech >= self._speech_start_chunks:
                    self._state = _State.SPEECH
                    self._segment_start = now - (self._consecutive_speech * self._chunk_duration_ms / 1000)
            else:
                self._consecutive_speech = 0
                self._speech_chunks.clear()
            return None

        if self._state == _State.SPEECH:
            self._speech_chunks.append(chunk.copy())
            if not vad_result.is_speech:
                self._consecutive_silence = 1
                self._state = _State.TRAILING_SILENCE
            if len(self._speech_chunks) >= self._max_chunks:
                return self._emit(now)
            return None

        if self._state == _State.TRAILING_SILENCE:
            self._speech_chunks.append(chunk.copy())
            if vad_result.is_speech:
                self._consecutive_silence = 0
                self._state = _State.SPEECH
                return None
            self._consecutive_silence += 1
            if self._consecutive_silence >= self._silence_end_chunks:
                return self._emit(now)
            if len(self._speech_chunks) >= self._max_chunks:
                return self._emit(now)
            return None

        return None  # unreachable

    def _emit(self, end_time: float) -> SpeechSegment:
        audio = np.concatenate(self._speech_chunks)
        duration = len(audio) / self._sample_rate
        # ponytail: minimum_speech_ms check — discard noise bursts
        min_dur = self._cfg.minimum_speech_ms / 1000
        self._reset()
        if duration < min_dur:
            return None  # type: ignore[return-value]
        self.segments_detected += 1
        return SpeechSegment(
            audio=audio,
            sample_rate=self._sample_rate,
            duration_seconds=duration,
            start_time=self._segment_start,
            end_time=end_time,
        )

    def _reset(self) -> None:
        self._state = _State.SILENCE
        self._speech_chunks.clear()
        self._consecutive_speech = 0
        self._consecutive_silence = 0
