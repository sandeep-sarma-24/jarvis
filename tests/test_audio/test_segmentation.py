import numpy as np
import pytest

from app.audio.segmentation import SpeechSegmenter
from app.audio.vad import VADResult
from app.config import VADConfig

CHUNK = 512
SR = 16000


@pytest.fixture
def segmenter():
    cfg = VADConfig(
        threshold=0.5,
        speech_end_silence_ms=96,   # 3 silence chunks × 32ms
        minimum_speech_ms=64,       # 2 chunks
        maximum_segment_seconds=1,
    )
    return SpeechSegmenter(config=cfg, sample_rate=SR, chunk_samples=CHUNK)


def _speech() -> VADResult:
    return VADResult(is_speech=True, confidence=0.9, latency_ms=0.1)


def _silence() -> VADResult:
    return VADResult(is_speech=False, confidence=0.1, latency_ms=0.1)


def _chunk() -> np.ndarray:
    return np.zeros(CHUNK, dtype=np.float32)


def test_speech_start_and_end(segmenter: SpeechSegmenter):
    for _ in range(5):
        result = segmenter.process(_chunk(), _speech())
    assert result is None

    segment = None
    for _ in range(5):
        r = segmenter.process(_chunk(), _silence())
        if r is not None:
            segment = r
    assert segment is not None
    assert segment.duration_seconds > 0


def test_short_utterance_discarded(segmenter: SpeechSegmenter):
    segmenter.process(_chunk(), _speech())
    for _ in range(4):
        result = segmenter.process(_chunk(), _silence())
    # 1 speech chunk = 32ms, below minimum_speech_ms=64ms
    assert result is None


def test_maximum_duration(segmenter: SpeechSegmenter):
    # 1 second max = 31.25 chunks → 31 chunks
    max_chunks = int(1 * SR / CHUNK)
    for i in range(max_chunks + 5):
        result = segmenter.process(_chunk(), _speech())
        if result is not None:
            break
    assert result is not None


def test_pause_does_not_split(segmenter: SpeechSegmenter):
    for _ in range(3):
        segmenter.process(_chunk(), _speech())

    # short pause (2 silence chunks, below end threshold of 3)
    segmenter.process(_chunk(), _silence())
    segmenter.process(_chunk(), _silence())

    for _ in range(3):
        r = segmenter.process(_chunk(), _speech())
    assert r is None  # still in speech, pause was too short

    segment = None
    for _ in range(5):
        r = segmenter.process(_chunk(), _silence())
        if r is not None:
            segment = r
    assert segment is not None


def test_accidental_single_frame_speech(segmenter: SpeechSegmenter):
    # single speech frame surrounded by silence shouldn't trigger
    segmenter.process(_chunk(), _speech())
    result = segmenter.process(_chunk(), _silence())
    assert result is None
    result = segmenter.process(_chunk(), _silence())
    assert result is None
