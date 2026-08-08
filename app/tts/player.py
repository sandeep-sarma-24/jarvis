"""Audio playback with interruption support.

Uses sounddevice for playback. sounddevice is already a project
dependency (used by microphone capture). Playback runs in a
background thread and can be stopped immediately.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class LocalAudioPlayer:
    """Non-blocking audio player with stop/interrupt capability.

    Playback policy: at most one response plays at a time.
    A new play() call stops any active playback first.
    """

    def __init__(self, *, volume: float = 1.0) -> None:
        self._volume = max(0.0, min(1.0, volume))
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._playing = threading.Event()
        self._stop_requested = threading.Event()
        self._playback_thread: threading.Thread | None = None

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Start playback. Stops any current playback first.

        Non-blocking: returns immediately, audio plays in background.
        """
        self.stop()

        self._stop_requested.clear()
        self._playing.set()

        self._playback_thread = threading.Thread(
            target=self._play_worker,
            args=(audio, sample_rate),
            daemon=True,
        )
        self._playback_thread.start()

    def stop(self) -> None:
        """Immediately stop active playback. Safe to call repeatedly."""
        self._stop_requested.set()

        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

        if self._playback_thread is not None and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None

        self._playing.clear()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def close(self) -> None:
        self.stop()

    def _play_worker(self, audio: np.ndarray, sample_rate: int) -> None:
        """Background worker that plays audio through sounddevice."""
        try:
            if self._volume < 1.0:
                audio = audio * self._volume

            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            block_size = 1024
            total_frames = audio.shape[0]

            with self._lock:
                if self._stop_requested.is_set():
                    return
                self._stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=audio.shape[1],
                    dtype="float32",
                    blocksize=block_size,
                )
                self._stream.start()

            offset = 0
            while offset < total_frames:
                if self._stop_requested.is_set():
                    break

                end = min(offset + block_size, total_frames)
                block = audio[offset:end]
                try:
                    with self._lock:
                        if self._stream is not None and self._stream.active:
                            self._stream.write(block)
                        else:
                            break
                except sd.PortAudioError:
                    break
                offset = end

        except Exception as exc:
            logger.error("Playback failed: %s", exc)
        finally:
            with self._lock:
                if self._stream is not None:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
            self._playing.clear()
