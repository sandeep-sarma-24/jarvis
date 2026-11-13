"""PhononTTS streaming behaviour with a fake ptts engine (no model, no audio)."""
import asyncio
import sys
import threading
import time
import types

import numpy as np
import pytest

from app.tts.local_tts import PhononTTS
from app.tts.model import StreamingTextToSpeech, TTSError

SR = 24000


class FakeStream:
    """Mimics ptts.AudioStream: an iterator with close()."""

    def __init__(self, n, *, gate=None, fail_at=None, delay=0.0):
        self.n, self.i, self.gate, self.fail_at, self.delay = n, 0, gate, fail_at, delay
        self.closed = False
        self.produced = 0
        self.sample_rate = SR

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or self.i >= self.n:
            raise StopIteration
        if self.gate is not None and self.i > 0:
            self.gate.wait(2)
        if self.fail_at == self.i:
            raise RuntimeError("decoder exploded")
        if self.delay:
            time.sleep(self.delay)
        self.i += 1
        self.produced += 1
        return np.full(2400, self.i, dtype=np.float32)

    def close(self):
        self.closed = True


class FakeModel:
    sample_rate = SR

    def __init__(self, **kw):
        self.kw = kw
        self.streams: list[FakeStream] = []

    def stream(self, text, *, voice=None):
        s = FakeStream(**self.kw)
        self.streams.append(s)
        return s


def _tts(**kw) -> tuple[PhononTTS, FakeModel]:
    tts = PhononTTS()
    tts._model = FakeModel(**kw)
    return tts, tts._model


def test_conforms_to_protocol():
    assert isinstance(PhononTTS(), StreamingTextToSpeech)


def test_chunks_in_order_and_stream_closed():
    tts, model = _tts(n=4)
    chunks = list(tts.synthesize_stream("hi"))
    assert [c.audio[0] for c in chunks] == [1, 2, 3, 4]
    assert all(c.sample_rate == SR and not c.is_last for c in chunks)
    assert model.streams[0].closed
    tts.close()


def test_first_chunk_without_lookahead():
    gate = threading.Event()  # blocks production of chunk 2
    tts, model = _tts(n=3, gate=gate)
    gen = tts.synthesize_stream("hi")
    first = next(gen)  # would hang if the generator held one chunk back
    assert first.audio[0] == 1 and model.streams[0].produced == 1
    gen.close()
    assert model.streams[0].closed
    tts.close()


def test_engine_failure_raises_tts_error():
    tts, model = _tts(n=3, fail_at=1)
    with pytest.raises(TTSError):
        list(tts.synthesize_stream("hi"))
    assert model.streams[0].closed
    tts.close()


def test_start_failure_raises_tts_error():
    tts = PhononTTS()
    tts._model = types.SimpleNamespace(stream=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(TTSError):
        list(tts.synthesize_stream("hi"))
    tts.close()


def test_cancel_stops_stream():
    tts, model = _tts(n=10)
    cancel = threading.Event()
    out = []
    for c in tts.synthesize_stream("hi", cancel=cancel):
        out.append(c)
        cancel.set()
    assert len(out) == 1 and model.streams[0].closed
    tts.close()


def test_stream_to_delivers_on_engine_thread():
    tts, _ = _tts(n=3)
    got, threads = [], set()

    def sink(c):
        got.append(c.audio[0])
        threads.add(threading.current_thread().name)
        return True

    stats = asyncio.run(tts.stream_to("hi", sink))
    assert got == [1, 2, 3]
    assert all(t.startswith("tts") for t in threads)
    assert stats.chunks == 3 and not stats.cancelled
    assert stats.audio_seconds == pytest.approx(3 * 2400 / SR)
    assert stats.first_chunk_ms is not None and stats.synth_ms >= stats.first_chunk_ms
    tts.close()


def test_stream_to_stops_when_sink_returns_false():
    tts, model = _tts(n=10)
    stats = asyncio.run(tts.stream_to("hi", lambda c: False))
    assert stats.chunks == 1 and stats.cancelled
    assert model.streams[0].produced == 1 and model.streams[0].closed
    tts.close()


def test_stream_to_stops_on_cancel_event():
    tts, model = _tts(n=10)
    cancel = threading.Event()

    def sink(c):
        cancel.set()
        return True

    stats = asyncio.run(tts.stream_to("hi", sink, cancel=cancel))
    assert stats.chunks == 1 and stats.cancelled and model.streams[0].closed
    tts.close()


def test_stream_to_raises_tts_error():
    tts, _ = _tts(n=3, fail_at=0)
    with pytest.raises(TTSError):
        asyncio.run(tts.stream_to("hi", lambda c: True))
    tts.close()


def test_task_cancellation_stops_worker():
    tts, model = _tts(n=1000, delay=0.01)

    async def main():
        task = asyncio.create_task(tts.stream_to("hi", lambda c: True))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    s = model.streams[0]
    deadline = time.monotonic() + 1
    while not s.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s.closed and s.produced < 50
    tts.close()


def test_close_while_streaming_stops_worker():
    tts, model = _tts(n=1000, delay=0.01)

    async def main():
        task = asyncio.create_task(tts.stream_to("hi", lambda c: True))
        await asyncio.sleep(0.05)
        tts.close()
        return await task

    stats = asyncio.run(main())
    assert stats.cancelled and model.streams[0].closed and stats.chunks < 50


def test_load_runs_on_executor_thread(monkeypatch):
    seen = {}

    class LoadModel(FakeModel):
        quant = "q8_0"

        def __init__(self, *a, **k):
            seen["thread"] = threading.current_thread().name
            super().__init__(n=1)

    monkeypatch.setitem(sys.modules, "ptts", types.SimpleNamespace(TTS=LoadModel))
    tts = PhononTTS()
    tts.load()
    tts.load()  # idempotent
    assert seen["thread"].startswith("tts")
    assert tts.sample_rate == SR
    tts.close()
