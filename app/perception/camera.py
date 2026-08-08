"""Camera abstraction and implementations.

Camera protocol defines a backend-agnostic capture interface.
OpenCVCamera provides a real device implementation.
FakeCamera provides deterministic frames for testing.

Lifecycle state machine: STOPPED → STARTING → RUNNING → STOPPING → STOPPED.
Camera failure is always isolated — never propagates beyond the perception subsystem.
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from app.perception.types import Frame

logger = logging.getLogger(__name__)


class CameraState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()


@runtime_checkable
class Camera(Protocol):
    @property
    def state(self) -> CameraState: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def capture(self) -> Frame | None: ...


class CameraError(Exception):
    """Non-fatal camera error — logged and suppressed by the pipeline."""


class OpenCVCamera:
    """Camera backed by OpenCV VideoCapture.

    The rest of JARVIS never sees cv2 objects — only Frame instances.
    Future sources (RTSP, USB, video file) plug in behind this same protocol.
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._state = CameraState.STOPPED
        self._cap = None  # cv2.VideoCapture, lazily imported

    @property
    def state(self) -> CameraState:
        return self._state

    async def start(self) -> None:
        if self._state != CameraState.STOPPED:
            return

        self._state = CameraState.STARTING
        try:
            import cv2

            cap = await asyncio.to_thread(cv2.VideoCapture, self._device_index)
            if not cap.isOpened():
                cap.release()
                raise CameraError(f"cannot open camera device {self._device_index}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap = cap
            self._state = CameraState.RUNNING
            logger.info("Camera started — device=%d %dx%d", self._device_index, self._width, self._height)
        except CameraError:
            self._state = CameraState.STOPPED
            raise
        except Exception as exc:
            self._state = CameraState.STOPPED
            raise CameraError(f"camera start failed: {exc}") from exc

    async def stop(self) -> None:
        if self._state not in (CameraState.RUNNING, CameraState.STARTING):
            return

        self._state = CameraState.STOPPING
        try:
            if self._cap is not None:
                await asyncio.to_thread(self._cap.release)
                self._cap = None
        finally:
            self._state = CameraState.STOPPED
            logger.info("Camera stopped")

    async def capture(self) -> Frame | None:
        if self._state != CameraState.RUNNING or self._cap is None:
            return None

        try:
            import cv2

            ret, frame = await asyncio.to_thread(self._cap.read)
            if not ret or frame is None:
                return None

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Frame(
                timestamp=time.time(),
                data=rgb.tobytes(),
                width=w,
                height=h,
                format="rgb24",
            )
        except Exception:
            logger.exception("Frame capture failed")
            return None


class FakeCamera:
    """Deterministic camera for testing — no hardware required.

    Generates frames with predictable byte patterns based on a counter.
    """

    def __init__(
        self,
        width: int = 4,
        height: int = 4,
        *,
        fail_on_start: bool = False,
        fail_on_capture: bool = False,
    ) -> None:
        self._width = width
        self._height = height
        self._state = CameraState.STOPPED
        self._frame_count = 0
        self._fail_on_start = fail_on_start
        self._fail_on_capture = fail_on_capture

    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def frame_count(self) -> int:
        return self._frame_count

    async def start(self) -> None:
        if self._state != CameraState.STOPPED:
            return

        self._state = CameraState.STARTING
        if self._fail_on_start:
            self._state = CameraState.STOPPED
            raise CameraError("fake camera start failure")

        self._state = CameraState.RUNNING

    async def stop(self) -> None:
        if self._state not in (CameraState.RUNNING, CameraState.STARTING):
            return
        self._state = CameraState.STOPPING
        self._state = CameraState.STOPPED

    async def capture(self) -> Frame | None:
        if self._state != CameraState.RUNNING:
            return None

        if self._fail_on_capture:
            return None

        self._frame_count += 1
        pixel_val = self._frame_count % 256
        data = bytes([pixel_val] * (self._width * self._height * 3))

        return Frame(
            timestamp=time.time(),
            data=data,
            width=self._width,
            height=self._height,
            format="rgb24",
        )
