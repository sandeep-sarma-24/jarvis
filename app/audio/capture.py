"""Microphone capture with bounded thread-safe ring buffer.

PortAudio callback (OS audio thread) → BoundedAudioBuffer → asyncio consumer.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from app.config import AudioConfig

logger = logging.getLogger(__name__)


class BoundedAudioBuffer:
    """Thread-safe bounded ring buffer bridging the PortAudio callback to asyncio.

    Overflow policy: drop oldest unprocessed chunk, increment counter, log warning.
    """

    def __init__(self, max_chunks: int) -> None:
        self._max = max_chunks
        self._buf: deque[np.ndarray] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.chunks_received = 0
        self.chunks_dropped = 0
        self.max_depth = 0

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._buf)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def put(self, chunk: np.ndarray) -> None:
        """Called from PortAudio thread — must not block."""
        with self._lock:
            self.chunks_received += 1
            if len(self._buf) == self._max:
                self.chunks_dropped += 1
            self._buf.append(chunk)
            current = len(self._buf)
            if current > self.max_depth:
                self.max_depth = current
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._event.set)

    def get_nowait(self) -> np.ndarray | None:
        with self._lock:
            if self._buf:
                return self._buf.popleft()
            return None

    async def get(self) -> np.ndarray:
        while True:
            chunk = self.get_nowait()
            if chunk is not None:
                return chunk
            self._event.clear()
            await self._event.wait()


class MicrophoneCapture:
    """Wraps sounddevice.InputStream with a bounded audio buffer."""

    def __init__(self, config: AudioConfig) -> None:
        self._cfg = config
        max_chunks = int(
            config.sample_rate * config.ring_buffer_seconds / config.chunk_samples
        )
        self.buffer = BoundedAudioBuffer(max_chunks)
        self._stream: sd.InputStream | None = None
        self._suppressed = False

    @property
    def suppressed(self) -> bool:
        return self._suppressed

    @suppressed.setter
    def suppressed(self, value: bool) -> None:
        self._suppressed = value

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if self._suppressed:
            return
        if status:
            logger.warning("sounddevice status: %s", status)
        self.buffer.put(indata[:, 0].copy())

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.buffer.bind_loop(loop)
        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype=self._cfg.dtype,
            blocksize=self._cfg.chunk_samples,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Microphone capture active — %d Hz, %d ch, %d-sample blocks",
            self._cfg.sample_rate,
            self._cfg.channels,
            self._cfg.chunk_samples,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Microphone capture stopped")
