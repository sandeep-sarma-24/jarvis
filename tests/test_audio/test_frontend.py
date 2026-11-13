"""AudioFrontend with fake mic, VAD and ASR (no models, no hardware)."""
import asyncio
import time

import numpy as np
import pytest

from app.audio.asr import ASRResult
from app.audio.capture import BoundedAudioBuffer
from app.audio.frontend import AudioFrontend
from app.audio.segmentation import SpeechSegmenter
from app.audio.vad import VADResult
from dataclasses import replace

from app.config import BargeInConfig, load_config

CHUNK = 512
SR = 16000
CHUNK_S = CHUNK / SR
SPEECH, QUIET, SILENT = 0.9, 0.6, 0.0  # QUIET is speech for VAD, below barge-in 0.8


class FakeMic:
    def __init__(self) -> None:
        self.buffer = BoundedAudioBuffer(max_chunks=1000)
        self.buffer.bind_loop(asyncio.get_running_loop())
        self.t = time.time()

    def feed(self, value: float, n: int = 1) -> None:
        for _ in range(n):
            self.t += CHUNK_S
            self.buffer.put(np.full(CHUNK, value, dtype=np.float32), self.t)

    def utterance(self, value: float = SPEECH, speech_chunks: int = 10) -> None:
        self.feed(value, speech_chunks)
        self.feed(SILENT, 6)


class FakeVAD:
    """Chunk fill value is the speech probability."""

    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def process_chunk(self, chunk):
        self.calls += 1
        p = float(chunk[0])
        return VADResult(is_speech=p >= 0.5, confidence=p, latency_ms=0.0)

    def reset(self) -> None:
        self.resets += 1


class FakeASR:
    def __init__(self, blocked: bool = False) -> None:
        self.gate = asyncio.Event()
        if not blocked:
            self.gate.set()
        self.seen: list[np.ndarray] = []

    async def transcribe_async(self, audio, sample_rate=16000):
        self.seen.append(audio)
        await self.gate.wait()
        return ASRResult(text=f"u{len(self.seen)}:{audio.max():.1f}", language="en", latency_ms=5.0)


async def _until(cond, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.001)


@pytest.fixture
async def rig():
    mic, vad, asr = FakeMic(), FakeVAD(), FakeASR()
    seg = SpeechSegmenter(replace(load_config().vad, short_utterance_ms=0), sample_rate=SR, chunk_samples=CHUNK)
    out, barges = [], []

    def make(barge_in: BargeInConfig = BargeInConfig(), tail_ms: int = 0, asr_=None):
        fe = AudioFrontend(mic, vad, seg, asr_ or asr, on_transcript=out.append,
                           on_barge_in=barges.append, barge_in=barge_in, playback_tail_ms=tail_ms)
        tasks.append(asyncio.create_task(fe.run()))
        return fe

    tasks: list[asyncio.Task] = []
    yield mic, vad, asr, make, out, barges
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_transcript_fields(rig):
    mic, vad, asr, make, out, _ = rig
    fe = make()
    mic.t = time.time() - 1.0  # captured in the past, as real audio is
    t_start = mic.t
    mic.utterance(speech_chunks=10)
    await _until(lambda: out)
    tr = out[0]
    assert tr.speech_start_time == pytest.approx(t_start)
    assert tr.speech_end_time == pytest.approx(t_start + 10 * CHUNK_S)
    assert tr.segment_emitted_time >= tr.speech_end_time
    assert tr.asr_latency_ms == 5.0
    assert tr.audio_seconds == pytest.approx(10 * CHUNK_S)  # no pre-roll available, silence trimmed
    assert "segments_dropped" in tr.metadata
    assert fe.metrics["speech_segments_transcribed"] == 1


async def test_keeps_consuming_during_slow_asr_and_drops_oldest(rig):
    mic, vad, _, make, out, _ = rig
    slow = FakeASR(blocked=True)
    fe = make(asr_=slow)
    for v in (0.91, 0.92, 0.93, 0.94):
        mic.utterance(v)
    # All four utterances are segmented while the first ASR call is still blocked.
    await _until(lambda: fe.metrics["speech_segments_detected"] == 4)
    assert len(slow.seen) == 1
    assert mic.buffer.depth == 0
    assert fe.metrics["speech_segments_dropped"] == 1  # queue of 2: 0.92 dropped
    slow.gate.set()
    await _until(lambda: len(out) == 3)
    assert [round(float(a.max()), 2) for a in slow.seen] == [0.91, 0.93, 0.94]


async def test_playback_suppression_resets_and_never_joins_across(rig):
    mic, vad, asr, make, out, barges = rig
    fe = make()
    mic.feed(SPEECH, 10)  # partial utterance, no endpoint yet
    await _until(lambda: vad.calls == 10)
    fe.set_playback_active(True)
    mic.feed(SPEECH, 20)  # assistant's own voice: discarded, no VAD
    await _until(lambda: mic.buffer.depth == 0)
    assert vad.calls == 10
    mic.feed(SPEECH, 5)  # still buffered when playback ends: drained
    fe.set_playback_active(False)
    assert mic.buffer.depth == 0
    mic.t = time.time()  # post-playback audio is captured after resume
    mic.utterance(0.7)
    await _until(lambda: out)
    await asyncio.sleep(0.01)
    assert len(out) == 1
    assert np.allclose(asr.seen[0], 0.7)  # nothing from before playback
    assert vad.resets == 2  # entering and leaving playback
    assert barges == []


async def test_playback_tail_discards_audio(rig):
    mic, vad, asr, make, out, _ = rig
    fe = make(tail_ms=200)
    fe.set_playback_active(True)
    fe.set_playback_active(False)
    mic.t = time.time()  # chunks within the 200 ms tail
    mic.feed(0.7, 4)
    mic.t = time.time() + 0.3
    mic.utterance(0.8)
    await _until(lambda: out)
    await asyncio.sleep(0.01)
    assert len(out) == 1 and np.allclose(asr.seen[0], 0.8)


async def test_barge_in_fires_once_when_enabled(rig):
    mic, vad, _, make, out, barges = rig
    fe = make(BargeInConfig(enabled=True, vad_threshold=0.8, min_speech_ms=256))
    fe.set_playback_active(True)
    mic.feed(QUIET, 20)  # below barge-in threshold
    mic.feed(SPEECH, 7)
    mic.feed(SILENT, 1)  # not sustained: counter restarts
    await _until(lambda: vad.calls == 28)
    assert barges == []
    t_before = mic.t
    mic.feed(SPEECH, 30)
    await _until(lambda: vad.calls == 58)
    assert barges == [pytest.approx(t_before + 8 * CHUNK_S)]
    assert out == [] and fe.metrics["speech_segments_detected"] == 0
    # A new playback period re-arms it.
    fe.set_playback_active(False)
    fe.set_playback_active(True)
    mic.feed(SPEECH, 8)
    await _until(lambda: len(barges) == 2)


async def test_barge_in_disabled_never_fires(rig):
    mic, vad, _, make, _, barges = rig
    fe = make(BargeInConfig(enabled=False))
    fe.set_playback_active(True)
    mic.feed(SPEECH, 30)
    await _until(lambda: mic.buffer.depth == 0)
    await asyncio.sleep(0.01)
    assert barges == [] and vad.calls == 0
    assert fe.metrics["chunks_discarded_playback"] == 30


async def test_asr_failure_does_not_stop_frontend(rig):
    mic, vad, _, make, out, _ = rig

    class Flaky(FakeASR):
        async def transcribe_async(self, audio, sample_rate=16000):
            if not self.seen:
                self.seen.append(audio)
                raise RuntimeError("boom")
            return await super().transcribe_async(audio, sample_rate)

    fe = make(asr_=Flaky())
    mic.utterance()
    mic.utterance()
    await _until(lambda: out)
    assert fe.metrics["speech_segments_failed"] == 1
