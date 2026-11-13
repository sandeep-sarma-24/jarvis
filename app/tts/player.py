"""Audio playback with interruption support.

StreamingPlayer is the streaming path (one persistent output stream, turn-tagged
queue). LocalAudioPlayer below is legacy and goes once StreamingPlayer is validated.

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
from collections import deque
from enum import Enum, auto
from typing import Callable

import numpy as np
import sounddevice as sd

from app.tts.model import AudioChunk

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


class StreamingPlayer:
    """One persistent callback-mode output stream fed by turn-tagged chunks.

    Implements app.tts.model.StreamingAudioPlayer. A chunk is played only if
    its turn is current when the device callback dequeues it. on_first_audio
    and on_done may fire on the PortAudio thread (or the caller of begin/end/
    stop) and must be cheap and non-blocking. on_done fires exactly once per
    begin(), with that turn's id, so callers can ignore stale completions.
    """

    def __init__(
        self,
        volume: float = 1.0,
        max_queue_seconds: float = 2.0,
        on_first_audio: Callable[[str], None] | None = None,
        on_done: Callable[[str, PlaybackOutcome], None] | None = None,
    ) -> None:
        self._volume = max(0.0, min(1.0, volume))
        self._max_queue_seconds = max_queue_seconds
        self._on_first_audio = on_first_audio
        self._on_done = on_done
        # Every critical section is short with no I/O, so the device callback
        # never waits long on it.
        self._cond = threading.Condition()
        self._stream: sd.OutputStream | None = None
        self._sample_rate: int | None = None
        self._max_samples = 0
        self._queue: deque[tuple[str, np.ndarray]] = deque()
        self._offset = 0  # samples of the head chunk already played
        self._queued = 0  # unplayed samples in the queue
        self._current: str | None = None  # turn whose audio may play
        self._pending: str | None = None  # turn still owed an on_done
        self._ended = False
        self._first_sent = False
        self._high_water = 0
        self._dropped = 0
        self._underruns = 0

    def start(self, sample_rate: int) -> None:
        if self._stream is not None:
            if sample_rate != self._sample_rate:
                raise ValueError(f"stream already open at {self._sample_rate} Hz")
            return
        with self._cond:
            self._sample_rate = sample_rate
            self._max_samples = int(self._max_queue_seconds * sample_rate)
        stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
            latency="low", callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def begin(self, turn_id: str) -> None:
        with self._cond:
            prev = self._pending
            self._clear_locked()
            self._current = self._pending = turn_id
            self._ended = False
            self._first_sent = False
        self._fire_done(prev, PlaybackOutcome.INTERRUPTED)

    def feed(self, chunk: AudioChunk, turn_id: str) -> bool:
        if self._sample_rate is None:
            raise RuntimeError("start() must be called before feed()")
        if chunk.sample_rate != self._sample_rate:
            raise ValueError(f"chunk at {chunk.sample_rate} Hz, stream at {self._sample_rate} Hz")
        audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1) * self._volume
        with self._cond:
            while True:
                if turn_id != self._current or self._ended:
                    self._dropped += 1
                    return False
                # An oversized chunk is admitted into an empty queue rather than deadlocking.
                if self._queued == 0 or self._queued + len(audio) <= self._max_samples:
                    break
                self._cond.wait()
            self._queue.append((turn_id, audio))
            self._queued += len(audio)
            self._high_water = max(self._high_water, self._queued)
            return True

    def end(self, turn_id: str) -> None:
        with self._cond:
            if turn_id != self._current:
                return
            self._ended = True
            done = self._take_done_locked()
        self._fire_done(done, PlaybackOutcome.COMPLETED)

    def stop(self) -> None:
        with self._cond:
            prev = self._pending
            self._clear_locked()
            self._current = self._pending = None
        self._fire_done(prev, PlaybackOutcome.INTERRUPTED)

    def is_playing(self) -> bool:
        """True from begin() until that turn's on_done."""
        with self._cond:
            return self._pending is not None

    @property
    def stats(self) -> dict[str, float]:
        sr = self._sample_rate or 1
        with self._cond:
            return {
                "queue_seconds": self._queued / sr,
                "max_queue_seconds": self._high_water / sr,
                "chunks_dropped_stale": float(self._dropped),
                "underruns": float(self._underruns),
            }

    def close(self) -> None:
        self.stop()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                logger.warning("Closing output stream failed: %s", exc)

    def _clear_locked(self) -> None:
        self._dropped += len(self._queue)
        self._queue.clear()
        self._queued = self._offset = 0
        self._cond.notify_all()  # wake blocked feeders so they re-check their turn

    def _take_done_locked(self) -> str | None:
        if self._ended and not self._queue and self._pending == self._current:
            done, self._pending = self._pending, None
            return done
        return None

    def _fire_done(self, turn_id: str | None, outcome: PlaybackOutcome) -> None:
        if turn_id is not None and self._on_done is not None:
            try:
                self._on_done(turn_id, outcome)
            except Exception:
                pass

    def _callback(self, outdata, frames, time_info, status) -> None:
        # PortAudio thread: no blocking calls, no logging.
        out = outdata[:, 0]
        filled = 0
        first = None
        with self._cond:
            while filled < frames and self._queue:
                tid, buf = self._queue[0]
                if tid != self._current:  # defence in depth; begin/stop already clear
                    self._queue.popleft()
                    self._queued -= len(buf) - self._offset
                    self._offset = 0
                    self._dropped += 1
                    continue
                n = min(frames - filled, len(buf) - self._offset)
                out[filled:filled + n] = buf[self._offset:self._offset + n]
                filled += n
                self._offset += n
                self._queued -= n
                if self._offset == len(buf):
                    self._queue.popleft()
                    self._offset = 0
            if filled:
                self._cond.notify_all()
                if not self._first_sent:
                    self._first_sent = True
                    first = self._current
            if filled < frames and self._current is not None and self._first_sent and not self._ended:
                # Only a gap after audio has started is audible as an underrun.
                self._underruns += 1
            done = self._take_done_locked()
        out[filled:] = 0
        if first is not None and self._on_first_audio is not None:
            try:
                self._on_first_audio(first)
            except Exception:
                pass
        self._fire_done(done, PlaybackOutcome.COMPLETED)
