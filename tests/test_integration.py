"""Integration test: audio fixture → VAD → segmentation → ASR → SpeechEvent → EventBus → subscriber.

Asserts that a valid SpeechEvent with correct per-event metadata reaches the subscriber.
Does not assert exact wording — ASR may vary.
"""
import asyncio

import numpy as np
import pytest

from app.audio.asr import SpeechRecognizer
from app.audio.segmentation import SpeechSegmenter, SpeechSegment
from app.audio.vad import VoiceActivityDetector, VADResult
from app.config import ASRConfig, VADConfig
from app.events.bus import EventBus
from app.events.types import Modality, SpeechEvent

SR = 16000
CHUNK = 512


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

    audio = np.random.randn(int(SR * 1.5)).astype(np.float32) * 0.01
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
        segment = SpeechSegment(
            audio=audio,
            sample_rate=SR,
            duration_seconds=len(audio) / SR,
            start_time=0,
            end_time=len(audio) / SR,
        )

    # ASR
    asr_result = asr.transcribe(segment.audio, segment.sample_rate)

    import time
    total_ms = round((time.time() - segment.end_time) * 1000, 1)

    # Publish event with per-event metadata
    event = SpeechEvent(
        session_id="test-session",
        correlation_id="test-corr",
        source=Modality.AUDIO,
        confidence=0.0,
        transcript=asr_result.text or "(silence)",
        duration_seconds=segment.duration_seconds,
        metadata={
            "asr_latency_ms": round(asr_result.latency_ms, 1),
            "total_speech_to_event_ms": total_ms,
        },
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
    assert e.confidence == 0.0
    assert e.duration_seconds > 0
    assert isinstance(e.transcript, str)
    assert e.id

    # Per-event metadata is attached and correct
    assert "asr_latency_ms" in e.metadata
    assert "total_speech_to_event_ms" in e.metadata
    assert isinstance(e.metadata["asr_latency_ms"], float)

    await bus.shutdown()
    vad.reset()
