import asyncio

import numpy as np
import pytest

from app.audio.capture import BoundedAudioBuffer


@pytest.fixture
def buffer():
    return BoundedAudioBuffer(max_chunks=5)


def test_put_and_get_nowait(buffer: BoundedAudioBuffer):
    chunk = np.ones(512, dtype=np.float32)
    buffer.put(chunk)
    got = buffer.get_nowait()
    assert got is not None
    np.testing.assert_array_equal(got, chunk)


def test_get_nowait_empty(buffer: BoundedAudioBuffer):
    assert buffer.get_nowait() is None


def test_bounded_capacity(buffer: BoundedAudioBuffer):
    for i in range(10):
        buffer.put(np.full(512, i, dtype=np.float32))
    assert buffer.chunks_received == 10
    assert buffer.chunks_dropped == 5

    values = []
    while True:
        c = buffer.get_nowait()
        if c is None:
            break
        values.append(c[0])
    assert values == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_no_unbounded_growth():
    buf = BoundedAudioBuffer(max_chunks=3)
    for _ in range(1000):
        buf.put(np.zeros(512, dtype=np.float32))
    count = 0
    while buf.get_nowait() is not None:
        count += 1
    assert count == 3


async def test_async_get():
    buf = BoundedAudioBuffer(max_chunks=5)
    loop = asyncio.get_running_loop()
    buf.bind_loop(loop)

    async def producer():
        await asyncio.sleep(0.02)
        buf.put(np.ones(512, dtype=np.float32))

    asyncio.create_task(producer())
    chunk = await asyncio.wait_for(buf.get(), timeout=1.0)
    assert chunk is not None
    assert chunk[0] == 1.0


def test_depth_empty(buffer: BoundedAudioBuffer):
    assert buffer.depth == 0


def test_depth_tracks_contents(buffer: BoundedAudioBuffer):
    buffer.put(np.zeros(512, dtype=np.float32))
    buffer.put(np.zeros(512, dtype=np.float32))
    assert buffer.depth == 2
    buffer.get_nowait()
    assert buffer.depth == 1


def test_max_depth_tracks_peak(buffer: BoundedAudioBuffer):
    for _ in range(3):
        buffer.put(np.zeros(512, dtype=np.float32))
    assert buffer.max_depth == 3
    buffer.get_nowait()
    buffer.get_nowait()
    assert buffer.max_depth == 3  # peak, not current


def test_overflow_tracks_dropped_count():
    buf = BoundedAudioBuffer(max_chunks=2)
    for _ in range(5):
        buf.put(np.zeros(512, dtype=np.float32))
    assert buf.chunks_dropped == 3
    assert buf.chunks_received == 5
    assert buf.max_depth == 2
