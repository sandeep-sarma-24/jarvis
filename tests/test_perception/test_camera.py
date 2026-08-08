"""Tests for camera abstraction, FakeCamera lifecycle, and failure isolation."""
from __future__ import annotations

import pytest

from app.perception.camera import CameraError, CameraState, FakeCamera


@pytest.mark.asyncio
async def test_fake_camera_lifecycle():
    cam = FakeCamera()
    assert cam.state == CameraState.STOPPED

    await cam.start()
    assert cam.state == CameraState.RUNNING

    frame = await cam.capture()
    assert frame is not None
    assert frame.width == 4
    assert frame.height == 4
    assert frame.format == "rgb24"
    assert len(frame.data) == 4 * 4 * 3

    await cam.stop()
    assert cam.state == CameraState.STOPPED


@pytest.mark.asyncio
async def test_fake_camera_frame_counter():
    cam = FakeCamera()
    await cam.start()

    for i in range(5):
        frame = await cam.capture()
        assert frame is not None
        assert cam.frame_count == i + 1

    await cam.stop()


@pytest.mark.asyncio
async def test_fake_camera_deterministic_frames():
    cam = FakeCamera(width=2, height=2)
    await cam.start()

    f1 = await cam.capture()
    f2 = await cam.capture()
    assert f1 is not None and f2 is not None
    assert f1.data != f2.data  # different pixel values (counter-based)
    assert f1.data[0] == 1  # frame_count=1 → pixel=1
    assert f2.data[0] == 2  # frame_count=2 → pixel=2

    await cam.stop()


@pytest.mark.asyncio
async def test_fake_camera_fail_on_start():
    cam = FakeCamera(fail_on_start=True)

    with pytest.raises(CameraError):
        await cam.start()

    assert cam.state == CameraState.STOPPED


@pytest.mark.asyncio
async def test_fake_camera_fail_on_capture():
    cam = FakeCamera(fail_on_capture=True)
    await cam.start()

    frame = await cam.capture()
    assert frame is None

    await cam.stop()


@pytest.mark.asyncio
async def test_capture_when_stopped_returns_none():
    cam = FakeCamera()
    frame = await cam.capture()
    assert frame is None


@pytest.mark.asyncio
async def test_double_start_is_noop():
    cam = FakeCamera()
    await cam.start()
    await cam.start()  # should not raise
    assert cam.state == CameraState.RUNNING
    await cam.stop()


@pytest.mark.asyncio
async def test_double_stop_is_noop():
    cam = FakeCamera()
    await cam.start()
    await cam.stop()
    await cam.stop()  # should not raise
    assert cam.state == CameraState.STOPPED
