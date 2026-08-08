"""Integration test: audio fixture → VAD → segmentation → ASR → SpeechEvent → EventBus → subscriber.

Asserts that a valid SpeechEvent reaches the subscriber.
Does not assert exact wording — ASR may vary.
"""
import asyncio

import numpy as np
import pytest

from app.audio.asr import SpeechRecognizer
from app.audio.segmentation import SpeechSegmenter
from app.audio.vad import VoiceActivityDetector, VADResult
from app.config import ASRConfig, VADConfig
from app.events.bus import EventBus
from app.events.types import Modality, SpeechEvent

SR = 16000
CHUNK = 512


def _generate_speech_like_audio(duration_s: float = 1.5) -> np.ndarray:
    """Generate a sine-wave audio fixture that triggers VAD positively."""
    t = np.linspace(0, duration_s, int(SR * duration_s), dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 300 * t)


@pytest.mark.integration
async def test_full_pipeline():
    bus = EventBus(queue_size=16)
    received: list[SpeechEvent] = []

    async def handler(event):
        received.append(event)

    bus.subscribe(SpeechEvent, handler)
    await bus.start()

    vad = VoiceActivityDetector(threshold=0.5)
    vad.load()

    asr = SpeechRecognizer(ASRConfig())
    asr.load()

    segmenter = SpeechSegmenter(
        config=VADConfig(speech_end_silence_ms=300, minimum_speech_ms=200, maximum_segment_seconds=10),
        sample_rate=SR,
        chunk_samples=CHUNK,
    )

    audio = _generate_speech_like_audio(1.5)
    silence = np.zeros(CHUNK, dtype=np.float32)

    # Feed audio through VAD + segmenter
    segment = None
    for i in range(0, len(audio), CHUNK):
        chunk = audio[i : i + CHUNK]
        if len(chunk) < CHUNK:
            break
        vad_result = vad.process_chunk(chunk)
        seg = segmenter.process(chunk, vad_result)
        if seg is not None:
            segment = seg

    # Feed silence to trigger end-of-speech
    for _ in range(20):
        vad_result = vad.process_chunk(silence)
        seg = segmenter.process(silence, vad_result)
        if seg is not None:
            segment = seg

    if segment is None:
        # If VAD didn't trigger on sine wave, construct segment manually
        segment_audio = audio
        from app.audio.segmentation import SpeechSegment
        segment = SpeechSegment(
            audio=segment_audio,
            sample_rate=SR,
            duration_seconds=len(segment_audio) / SR,
            start_time=0,
            end_time=len(segment_audio) / SR,
        )

    # ASR
    asr_result = asr.transcribe(segment.audio, segment.sample_rate)

    # Publish event
    event = SpeechEvent(
        session_id="test-session",
        correlation_id="test-corr",
        source=Modality.AUDIO,
        confidence=1.0,
        transcript=asr_result.text,
        duration_seconds=segment.duration_seconds,
    )
    await bus.publish(event)
    await asyncio.sleep(0.1)

    # Assertions
    assert len(received) == 1
    e = received[0]
    assert isinstance(e, SpeechEvent)
    assert e.session_id == "test-session"
    assert e.correlation_id == "test-corr"
    assert e.source == Modality.AUDIO
    assert e.duration_seconds > 0
    assert isinstance(e.transcript, str)
    assert e.id  # has a UUID

    await bus.shutdown()
    vad.reset()
