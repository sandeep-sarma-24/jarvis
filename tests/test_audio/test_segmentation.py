import math

import numpy as np
import pytest

from app.audio.segmentation import SpeechSegmenter, SpeechSegment
from app.audio.vad import VADResult
from app.config import VADConfig

CHUNK = 512
SR = 16000
CHUNK_MS = (CHUNK / SR) * 1000  # 32ms


@pytest.fixture
def segmenter():
    cfg = VADConfig(
        threshold=0.5,
        speech_end_silence_ms=96,   # ceil(96/32) = 3 silence chunks
        minimum_speech_ms=64,       # ceil(64/32) = 2 chunks
        maximum_segment_seconds=1,
    )
    return SpeechSegmenter(config=cfg, sample_rate=SR, chunk_samples=CHUNK)


def _speech() -> VADResult:
    return VADResult(is_speech=True, confidence=0.9, latency_ms=0.1)


def _silence() -> VADResult:
    return VADResult(is_speech=False, confidence=0.1, latency_ms=0.1)


def _chunk() -> np.ndarray:
    return np.zeros(CHUNK, dtype=np.float32)


def _collect(segmenter: SpeechSegmenter, vad_results: list[VADResult]) -> list[SpeechSegment]:
    """Feed a sequence of VAD results and return all emitted segments."""
    segments = []
    for vr in vad_results:
        seg = segmenter.process(_chunk(), vr)
        if seg is not None:
            segments.append(seg)
    return segments


# ── Basic speech start/end ──────────────────────────────────────────────

def test_speech_start_and_end(segmenter: SpeechSegmenter):
    results = [_speech()] * 5 + [_silence()] * 5
    segments = _collect(segmenter, results)
    assert len(segments) == 1
    assert segments[0].duration_seconds > 0


# ── Minimum speech duration ─────────────────────────────────────────────

def test_exact_minimum_boundary(segmenter: SpeechSegmenter):
    """Exactly minimum_speech_ms (2 chunks = 64ms) should produce a segment."""
    results = [_speech()] * 2 + [_silence()] * 5
    segments = _collect(segmenter, results)
    assert len(segments) == 1


def test_below_minimum_speech_discarded(segmenter: SpeechSegmenter):
    """1 speech chunk = 32ms, below minimum_speech_ms = 64ms."""
    results = [_speech()] * 1 + [_silence()] * 5
    segments = _collect(segmenter, results)
    assert len(segments) == 0


def test_short_utterance_discarded(segmenter: SpeechSegmenter):
    segmenter.process(_chunk(), _speech())
    for _ in range(4):
        segmenter.process(_chunk(), _silence())
    # state should have reset
    assert segmenter._state.name == "SILENCE"


# ── Trailing silence ────────────────────────────────────────────────────

def test_trailing_silence_exact_threshold(segmenter: SpeechSegmenter):
    """Exactly 3 silence chunks (=96ms threshold) should finalize."""
    results = [_speech()] * 3 + [_silence()] * 3
    segments = _collect(segmenter, results)
    assert len(segments) == 1


def test_trailing_silence_below_threshold_continues(segmenter: SpeechSegmenter):
    """2 silence chunks (<3 threshold) should not finalize."""
    results = [_speech()] * 3 + [_silence()] * 2
    segments = _collect(segmenter, results)
    assert len(segments) == 0


# ── Pause handling ──────────────────────────────────────────────────────

def test_pause_shorter_than_threshold(segmenter: SpeechSegmenter):
    """Short pause (2 chunks) should not split into separate segments."""
    results = [_speech()] * 3 + [_silence()] * 2 + [_speech()] * 3 + [_silence()] * 5
    segments = _collect(segmenter, results)
    assert len(segments) == 1


def test_pause_at_threshold_finalizes(segmenter: SpeechSegmenter):
    """Pause of exactly 3 silence chunks should finalize the first segment."""
    results = [_speech()] * 3 + [_silence()] * 3 + [_speech()] * 3 + [_silence()] * 5
    segments = _collect(segmenter, results)
    assert len(segments) == 2


# ── Maximum segment duration ───────────────────────────────────────────

def test_maximum_duration(segmenter: SpeechSegmenter):
    max_chunks = int(1 * SR / CHUNK)
    results = [_speech()] * (max_chunks + 5)
    segments = _collect(segmenter, results)
    assert len(segments) >= 1


# ── Accidental single-frame noise ──────────────────────────────────────

def test_accidental_single_frame_speech(segmenter: SpeechSegmenter):
    results = [_speech()] + [_silence()] * 3
    segments = _collect(segmenter, results)
    assert len(segments) == 0


# ── Chunk conversion math ─────────────────────────────────────────────

def test_chunk_conversion_uses_ceil():
    cfg = VADConfig(
        speech_end_silence_ms=50,   # 50/32 = 1.5625 → ceil = 2
        minimum_speech_ms=50,       # 50/32 = 1.5625 → ceil = 2
        maximum_segment_seconds=5,
    )
    seg = SpeechSegmenter(config=cfg, sample_rate=SR, chunk_samples=CHUNK)
    assert seg._silence_end_chunks == math.ceil(50 / CHUNK_MS)
    assert seg._min_speech_chunks == math.ceil(50 / CHUNK_MS)


# ── Return type correctness ───────────────────────────────────────────

def test_emit_returns_none_for_too_short():
    cfg = VADConfig(minimum_speech_ms=500, speech_end_silence_ms=96, maximum_segment_seconds=5)
    seg = SpeechSegmenter(config=cfg, sample_rate=SR, chunk_samples=CHUNK)
    # 3 speech chunks = 96ms, well below 500ms minimum
    results = [_speech()] * 3 + [_silence()] * 5
    segments = _collect(seg, results)
    assert len(segments) == 0
