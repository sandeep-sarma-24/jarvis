"""Speech segmentation — state machine that groups VAD results into utterances.

Responsibility: "Does this sequence of VAD results form one utterance?"
Consumes VADResult stream, emits SpeechSegment when an utterance is complete.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
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
    """A completed utterance ready for ASR (pre-roll included, trailing silence trimmed).

    Times are capture times (time.time() seconds). end_time is the capture time
    of the chunk that triggered endpointing; speech_end_time is that of the last
    chunk classified as speech.
    """

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    start_time: float
    end_time: float
    speech_end_time: float | None = None

    def __post_init__(self) -> None:
        if self.speech_end_time is None:
            self.speech_end_time = self.end_time


class SpeechSegmenter:
    """State machine: SILENCE → SPEECH → TRAILING_SILENCE → emit segment."""

    def __init__(self, config: VADConfig, sample_rate: int = 16000, chunk_samples: int = 512) -> None:
        self._cfg = config
        self.sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._chunk_duration_ms = (chunk_samples / sample_rate) * 1000

        self._min_speech_chunks = math.ceil(config.minimum_speech_ms / self._chunk_duration_ms)
        self._silence_end_chunks = math.ceil(config.speech_end_silence_ms / self._chunk_duration_ms)
        self._max_chunks = int(config.maximum_segment_seconds * sample_rate / chunk_samples)
        pre_roll_chunks = math.ceil(config.pre_roll_ms / self._chunk_duration_ms)

        self._state = _State.SILENCE
        # In SILENCE: the last pre_roll chunks plus up to 2 onset candidates.
        self._recent: deque[np.ndarray] = deque(maxlen=pre_roll_chunks + 2)
        self._chunks: list[np.ndarray] = []
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._speech_count = 0
        self._last_speech_index = 0
        self._segment_start = 0.0
        self._speech_end = 0.0

        self.segments_detected = 0

    def process(
        self, chunk: np.ndarray, vad_result: VADResult, capture_time: float | None = None
    ) -> SpeechSegment | None:
        """Feed one chunk + its VAD result. Returns a SpeechSegment when complete, else None.

        capture_time is when the chunk's last sample was captured; defaults to now.
        """
        t = time.time() if capture_time is None else capture_time

        if self._state == _State.SILENCE:
            self._recent.append(chunk.copy())
            if not vad_result.is_speech:
                self._consecutive_speech = 0
                return None
            self._consecutive_speech += 1
            if self._consecutive_speech == 1:
                self._segment_start = t - self._chunk_duration_ms / 1000
            elif self._consecutive_speech >= 2:
                self._state = _State.SPEECH
                self._chunks = list(self._recent)
                self._recent.clear()
                self._speech_count = 2
                self._last_speech_index = len(self._chunks) - 1
                self._speech_end = t
            return None

        self._chunks.append(chunk.copy())
        if vad_result.is_speech:
            self._speech_count += 1
            self._last_speech_index = len(self._chunks) - 1
            self._speech_end = t
            self._consecutive_silence = 0
            self._state = _State.SPEECH
        else:
            self._consecutive_silence += 1
            self._state = _State.TRAILING_SILENCE
            if self._consecutive_silence >= self._silence_end_chunks:
                return self._emit(t)
        if len(self._chunks) >= self._max_chunks:
            return self._emit(t)
        return None

    def _emit(self, end_time: float) -> SpeechSegment | None:
        speech_count = self._speech_count
        audio = np.concatenate(self._chunks[: self._last_speech_index + 1])
        start, speech_end = self._segment_start, self._speech_end
        self.reset()
        # Counts speech chunks only: a click plus trailing silence must not pass.
        if speech_count < self._min_speech_chunks:
            return None
        self.segments_detected += 1
        return SpeechSegment(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=len(audio) / self.sample_rate,
            start_time=start,
            end_time=end_time,
            speech_end_time=speech_end,
        )

    def reset(self) -> None:
        """Forget any partial utterance and pre-roll."""
        self._state = _State.SILENCE
        self._recent.clear()
        self._chunks = []
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._speech_count = 0
