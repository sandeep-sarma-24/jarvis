"""Audio playback with interruption support.

Uses sounddevice for playback. sounddevice is already a project
dependency (used by microphone capture). Playback runs in a
background thread and can be stopped immediately.

Generation-based cancellation: each play() increments a generation counter.
The worker thread exits if its generation is no longer current, preventing
overlapping playback or resurrection of old workers.
"""
from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class PlaybackOutcome(Enum):
    COMPLETED = auto()
    INTERRUPTED = auto()


OnPlaybackDone = Callable[[PlaybackOutcome], None]


class LocalAudioPlayer:
    """Non-blocking audio player with stop/interrupt capability.

    Playback policy: at most one response plays at a time.
    A new play() call stops any active playback first.

    Generation-based cancellation ensures no overlapping streams:
    - Each play() increments _generation.
    - The worker thread holds its own generation value.
    - If the worker's generation != current _generation, it exits as INTERRUPTED.
    - stop() increments generation and aborts the stream.

    Completion semantics:
    - COMPLETED: all audio frames written, generation still current at end.
    - INTERRUPTED: anything else (stale generation, stop(), device error).
    """

    def __init__(
        self,
        *,
        volume: float = 1.0,
        on_done: OnPlaybackDone | None = None,
    ) -> None:
        self._volume = max(0.0, min(1.0, volume))
        self._on_done = on_done
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._generation = 0
        self._playing = False
        self._worker_thread: threading.Thread | None = None

    @property
    def on_done(self) -> OnPlaybackDone | None:
        return self._on_done

    @on_done.setter
    def on_done(self, cb: OnPlaybackDone | None) -> None:
        self._on_done = cb

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Start playback. Stops any current playback first.

        Non-blocking: returns immediately, audio plays in background.
        """
        self._cancel_current()

        with self._lock:
            self._generation += 1
            gen = self._generation
            self._playing = True

        self._worker_thread = threading.Thread(
            target=self._play_worker,
            args=(audio, sample_rate, gen),
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Immediately stop active playback. Safe to call repeatedly.

        After stop() returns:
        - is_playing() == False
        - No active playback stream remains
        - The worker thread is joined or confirmed dead
        """
        self._cancel_current()

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def close(self) -> None:
        self.stop()

    def _cancel_current(self) -> None:
        """Invalidate current generation and tear down stream."""
        with self._lock:
            self._generation += 1
            self._playing = False
            if self._stream is not None:
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._worker_thread = None

    def _is_current(self, gen: int) -> bool:
        with self._lock:
            return self._generation == gen

    def _play_worker(self, audio: np.ndarray, sample_rate: int, gen: int) -> None:
        """Background worker that plays audio through sounddevice.

        Exits immediately if its generation is no longer current.
        Only reports COMPLETED when ALL frames were written and generation
        remained current throughout. Every other exit path is INTERRUPTED.
        """
        completed = False
        try:
            if not self._is_current(gen):
                return

            if self._volume < 1.0:
                audio = audio * self._volume

            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            block_size = 1024
            total_frames = audio.shape[0]

            with self._lock:
                if self._generation != gen:
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
                if not self._is_current(gen):
                    break

                end = min(offset + block_size, total_frames)
                block = audio[offset:end]
                try:
                    with self._lock:
                        if self._stream is not None and self._stream.active and self._generation == gen:
                            self._stream.write(block)
                        else:
                            break
                except sd.PortAudioError as exc:
                    logger.error("Playback device error: %s", exc)
                    break
                offset = end

            # Only mark completed if we wrote all frames and still own the generation
            if offset >= total_frames and self._is_current(gen):
                completed = True

        except Exception as exc:
            logger.error("Playback failed: %s", exc)
        finally:
            with self._lock:
                if self._generation == gen:
                    if self._stream is not None:
                        try:
                            self._stream.stop()
                            self._stream.close()
                        except Exception:
                            pass
                        self._stream = None
                    self._playing = False

            outcome = PlaybackOutcome.COMPLETED if completed else PlaybackOutcome.INTERRUPTED
            if self._on_done is not None:
                try:
                    self._on_done(outcome)
                except Exception:
                    pass
