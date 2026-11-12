"""Hardware-free fakes for the end-to-end benchmark and stability test.

They implement the frozen protocols in app/reasoning/model.py and app/tts/model.py
with configurable timing, so the real orchestration code can be timed without a
microphone, speaker, model or network. Timings measured with them are SIMULATED.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import random
import threading
import time
from collections import deque
from typing import Callable, Iterator

import numpy as np

from app.contracts import LatencyRecorder, Transcript, Turn
from app.reasoning.model import ReasoningError, ReasoningResult
from app.tts.model import AudioChunk, TTSResult, TTSStreamStats
from app.tts.player import PlaybackOutcome

RESPONSE = "Certainly. The build finished without errors. Anything else you need?"


def _wait(cancel: threading.Event | None, seconds: float) -> bool:
    """Sleep for `seconds` or until cancelled. True if cancelled."""
    if cancel is None:
        time.sleep(seconds)
        return False
    return cancel.wait(seconds)


def make_transcript(text: str, speech_end_time: float, audio_seconds: float = 1.5) -> Transcript:
    return Transcript(
        text=text,
        speech_start_time=speech_end_time - audio_seconds,
        speech_end_time=speech_end_time,
        segment_emitted_time=speech_end_time,
        asr_latency_ms=0.0,
        audio_seconds=audio_seconds,
    )


class CountingRecorder(LatencyRecorder):
    """LatencyRecorder that also counts records past its maxlen."""

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__(maxlen)
        self.count = 0

    def record(self, turn: Turn, outcome: str) -> dict[str, object]:
        self.count += 1
        return super().record(turn, outcome)


class FakeStreamingModel:
    """StreamingReasoningModel with a fixed answer, time-to-first-token and token rate."""

    def __init__(self, *, ttft_ms: float = 150.0, tokens_per_second: float = 40.0,
                 fail_rate: float = 0.0, text: str = RESPONSE, seed: int = 0) -> None:
        self.ttft_s = ttft_ms / 1000
        self.tps = tokens_per_second
        self.fail_rate = fail_rate
        self.tokens = [w + " " for w in text.split()]
        self._rng = random.Random(seed)
        self.last_stats: dict[str, float] = {}

    @property
    def model_name(self) -> str:
        return "fake-stream"

    def generate(self, messages, *, max_tokens: int = 256, temperature: float = 0.2) -> ReasoningResult:
        t0 = time.perf_counter()
        n = min(max_tokens, len(self.tokens))
        time.sleep(self.ttft_s + n / self.tps)
        return ReasoningResult(text="".join(self.tokens[:n]).strip(), model=self.model_name,
                               latency_ms=(time.perf_counter() - t0) * 1000, success=True)

    def stream(self, messages, *, max_tokens: int = 256, temperature: float = 0.2,
               cancel: threading.Event | None = None) -> Iterator[str]:
        if self._rng.random() < self.fail_rate:
            raise ReasoningError("injected failure")
        if _wait(cancel, self.ttft_s):
            return
        for i, tok in enumerate(self.tokens[:max_tokens]):
            if i and _wait(cancel, 1 / self.tps):
                return
            yield tok
        self.last_stats = {"ttft_ms": self.ttft_s * 1000, "output_tokens": i + 1,
                           "tokens_per_second": self.tps}

    def warm_up(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeTTS:
    """StreamingTextToSpeech producing silence with a first-chunk delay and real-time factor."""

    def __init__(self, *, first_chunk_ms: float = 80.0, rtf: float = 0.2, sample_rate: int = 24000,
                 chunk_seconds: float = 0.1, seconds_per_char: float = 0.06) -> None:
        self.first_chunk_s = first_chunk_ms / 1000
        self.rtf = rtf
        self._sr = sample_rate
        self.chunk_seconds = chunk_seconds
        self.seconds_per_char = seconds_per_char
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake-tts")

    @property
    def model_name(self) -> str:
        return "fake-tts"

    @property
    def sample_rate(self) -> int:
        return self._sr

    def load(self) -> None:
        pass

    def _chunks(self, text: str) -> int:
        return max(1, math.ceil(len(text) * self.seconds_per_char / self.chunk_seconds))

    def synthesize(self, text: str) -> TTSResult:
        n = self._chunks(text)
        time.sleep(self.first_chunk_s + n * self.chunk_seconds * self.rtf)
        audio = np.zeros(int(n * self.chunk_seconds * self._sr), dtype=np.float32)
        return TTSResult(True, audio, self._sr, 1, len(audio) / self._sr, 0.0, self.model_name)

    def synthesize_stream(self, text: str) -> Iterator[AudioChunk]:
        yield AudioChunk(self.synthesize(text).audio, self._sr, is_last=True)

    def _run(self, text: str, sink: Callable[[AudioChunk], bool], cancel: threading.Event | None) -> TTSStreamStats:
        t0 = time.perf_counter()
        n = self._chunks(text)
        first = None
        produced = 0
        cancelled = _wait(cancel, self.first_chunk_s)
        while not cancelled and produced < n:
            if produced and _wait(cancel, self.chunk_seconds * self.rtf):
                cancelled = True
                break
            chunk = AudioChunk(np.zeros(int(self.chunk_seconds * self._sr), dtype=np.float32),
                               self._sr, is_last=produced == n - 1)
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            produced += 1
            if not sink(chunk):
                cancelled = True
        return TTSStreamStats(first_chunk_ms=first, synth_ms=(time.perf_counter() - t0) * 1000,
                              audio_seconds=produced * self.chunk_seconds, chunks=produced,
                              cancelled=cancelled)

    async def stream_to(self, text: str, sink: Callable[[AudioChunk], bool], *,
                        cancel: threading.Event | None = None) -> TTSStreamStats:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run, text, sink, cancel)

    def close(self) -> None:
        self._executor.shutdown(wait=False)


class FakePlayer:
    """StreamingAudioPlayer that consumes audio in real time on its own thread, without a device.

    It also records, per turn, when the first and last device block started, so the
    benchmark can measure first audio and cancellation independently of TurnManager.
    """

    def __init__(self, volume: float = 1.0, max_queue_seconds: float = 2.0,
                 on_first_audio: Callable[[str], None] | None = None,
                 on_done: Callable[[str, PlaybackOutcome], None] | None = None,
                 block_seconds: float = 0.01) -> None:
        self.max_queue_s = max_queue_seconds
        self.on_first_audio = on_first_audio
        self.on_done = on_done
        self.block_s = block_seconds
        self._cond = threading.Condition(threading.RLock())
        self._queue: deque[tuple[str, np.ndarray]] = deque()
        self._queued = 0  # samples
        self._sr = 0
        self._current: str | None = None
        self._ended = False
        self._done_fired = True
        self._playing = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._stopped: deque[str] = deque(maxlen=64)  # turns interrupted by stop() or a newer begin()
        # shortcut: per-turn dicts grow by ~200 bytes a turn; fine for benchmark-sized runs
        self.begins: list[str] = []
        self.first_audio: dict[str, float] = {}
        self.last_block: dict[str, float] = {}
        self.done: dict[str, PlaybackOutcome] = {}
        self.stop_time: dict[str, float] = {}
        self.stale_played = 0
        self._stats = {"queue_seconds": 0.0, "max_queue_seconds": 0.0,
                       "chunks_dropped_stale": 0.0, "underruns": 0.0}

    def start(self, sample_rate: int) -> None:
        self._sr = sample_rate
        self._thread = threading.Thread(target=self._device, name="fake-player", daemon=True)
        self._thread.start()

    def _finish(self, outcome: PlaybackOutcome) -> None:
        """Fire on_done once for the current turn. Caller holds the lock."""
        if self._current is None or self._done_fired:
            return
        self._done_fired = True
        self.done[self._current] = outcome
        if self.on_done is not None:
            self.on_done(self._current, outcome)

    def _drop_queue(self) -> None:
        self._stats["chunks_dropped_stale"] += len(self._queue)
        self._queue.clear()
        self._queued = 0
        self._stats["queue_seconds"] = 0.0

    def begin(self, turn_id: str) -> None:
        with self._cond:
            if self._current is not None and self._current != turn_id and not self._done_fired:
                self._stopped.append(self._current)
                self._finish(PlaybackOutcome.INTERRUPTED)
            self._drop_queue()
            self._current, self._ended, self._done_fired = turn_id, False, False
            self.begins.append(turn_id)
            self._cond.notify_all()

    def feed(self, chunk: AudioChunk, turn_id: str) -> bool:
        audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
        with self._cond:
            while (turn_id == self._current and not self._closed and self._queue
                   and (self._queued + audio.size) / self._sr > self.max_queue_s):
                self._cond.wait(0.05)
            if turn_id != self._current or self._closed:
                return False
            self._queue.append((turn_id, audio))
            self._queued += audio.size
            q = self._queued / self._sr
            self._stats["queue_seconds"] = q
            self._stats["max_queue_seconds"] = max(self._stats["max_queue_seconds"], q)
            self._cond.notify_all()
            return True

    def end(self, turn_id: str) -> None:
        with self._cond:
            if turn_id == self._current:
                self._ended = True
                self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            if self._current is not None:
                self._stopped.append(self._current)
                self.stop_time.setdefault(self._current, time.time())
                self._finish(PlaybackOutcome.INTERRUPTED)
            self._drop_queue()
            self._current = None
            self._cond.notify_all()

    def is_playing(self) -> bool:
        return self._playing

    @property
    def stats(self) -> dict[str, float]:
        return dict(self._stats)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _device(self) -> None:
        block = max(1, int(self._sr * self.block_s))
        while True:
            with self._cond:
                if self._closed:
                    return
                if not self._queue:
                    self._playing = False
                    if self._ended:
                        self._finish(PlaybackOutcome.COMPLETED)
                    self._cond.wait(self.block_s)
                    continue
                tid, audio = self._queue[0]
                n = min(block, audio.size)
                if n == audio.size:
                    self._queue.popleft()
                else:
                    self._queue[0] = (tid, audio[n:])
                self._queued -= n
                self._stats["queue_seconds"] = self._queued / self._sr
                now = time.time()
                if tid in self._stopped:
                    self.stale_played += 1
                self._playing = True
                if tid not in self.first_audio:
                    self.first_audio[tid] = now
                    if self.on_first_audio is not None:
                        self.on_first_audio(tid)
                self.last_block[tid] = now
                self._cond.notify_all()
            time.sleep(n / self._sr)
