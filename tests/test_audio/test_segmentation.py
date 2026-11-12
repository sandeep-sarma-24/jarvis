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


# ── Default config (configs/default.toml) ─────────────────────────────

def _default_segmenter() -> SpeechSegmenter:
    from app.config import load_config

    return SpeechSegmenter(config=load_config().vad, sample_rate=SR, chunk_samples=CHUNK)


def _feed(seg: SpeechSegmenter, pattern: list[tuple[float, bool]], t0: float = 1000.0):
    """pattern: (chunk fill value, is_speech). Chunk i is captured at t0 + (i+1)*32 ms."""
    out = []
    for i, (value, speech) in enumerate(pattern):
        vr = _speech() if speech else _silence()
        s = seg.process(np.full(CHUNK, value, dtype=np.float32), vr, t0 + (i + 1) * CHUNK / SR)
        if s is not None:
            out.append(s)
    return out


def test_default_endpoint_delay_is_six_chunks():
    seg = _default_segmenter()
    assert seg._silence_end_chunks == 6
    segments = _feed(seg, [(1.0, True)] * 10 + [(0.0, False)] * 6)
    assert len(segments) == 1
    s = segments[0]
    assert s.end_time - s.speech_end_time == pytest.approx(6 * CHUNK / SR)
    # 5 silence chunks are not enough
    assert _feed(_default_segmenter(), [(1.0, True)] * 10 + [(0.0, False)] * 5) == []


def test_default_minimum_speech_counts_speech_chunks_only():
    # 64 ms click followed by a full endpoint of silence: must not reach ASR
    assert _feed(_default_segmenter(), [(1.0, True)] * 2 + [(0.0, False)] * 20) == []
    # 250 ms → 8 speech chunks is the minimum
    assert _feed(_default_segmenter(), [(1.0, True)] * 7 + [(0.0, False)] * 6) == []
    assert len(_feed(_default_segmenter(), [(1.0, True)] * 8 + [(0.0, False)] * 6)) == 1


def test_speech_end_time_is_capture_time_of_last_speech_chunk():
    t0 = 5000.0
    pattern = [(0.0, False)] * 4 + [(1.0, True)] * 10 + [(0.0, False)] * 2 + [(1.0, True)] * 3 + [(0.0, False)] * 6
    s = _feed(_default_segmenter(), pattern, t0)[0]
    last_speech = 4 + 10 + 2 + 3  # 1-based index of the last speech chunk
    assert s.speech_end_time == pytest.approx(t0 + last_speech * CHUNK / SR)
    assert s.start_time == pytest.approx(t0 + 4 * CHUNK / SR)  # start of first speech chunk
    assert s.end_time == pytest.approx(t0 + len(pattern) * CHUNK / SR)


def test_pre_roll_kept_and_trailing_silence_trimmed():
    # pre-roll 96 ms = 3 chunks; values tag where each chunk came from
    pattern = [(0.1, False)] * 2 + [(0.2, False)] * 3 + [(1.0, True)] * 10 + [(0.3, False)] * 6
    s = _feed(_default_segmenter(), pattern)[0]
    values = s.audio.reshape(-1, CHUNK)[:, 0]
    assert list(values[:3]) == pytest.approx([0.2] * 3)  # exactly pre_roll before onset
    assert list(values[3:]) == pytest.approx([1.0] * 10)  # no trailing silence
    assert s.duration_seconds == pytest.approx(13 * CHUNK / SR)


def test_reset_drops_partial_utterance():
    seg = _default_segmenter()
    _feed(seg, [(1.0, True)] * 10)
    seg.reset()
    s = _feed(seg, [(2.0, True)] * 8 + [(0.0, False)] * 6)[0]
    assert set(s.audio.reshape(-1, CHUNK)[:, 0]) == {2.0}


def test_segment_constructor_without_speech_end_time():
    s = SpeechSegment(audio=np.zeros(1), sample_rate=SR, duration_seconds=0, start_time=1, end_time=2)
    assert s.speech_end_time == 2
