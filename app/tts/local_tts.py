"""Local TTS implementation using Gradium Phonon 31m (English) via ptts.

Runs entirely on-device (Rust runtime, no PyTorch). No network required
after initial model download (~65 MB q8 GGUF from HuggingFace Hub).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from typing import Callable, Iterator

import numpy as np

from app.tts.model import AudioChunk, TTSError, TTSResult, TTSStreamStats

logger = logging.getLogger(__name__)


class PhononTTS:
    """Phonon 31m English TTS via the ptts package. Implements StreamingTextToSpeech."""

    def __init__(
        self,
        *,
        model_id: str = "hf://gradium/phonon-31m-en",
        voice: str = "Easton",
        quant: str = "q8_0",
    ) -> None:
        self._model_id = model_id
        self._voice = voice
        self._quant = quant
        self._sample_rate = 24000  # overwritten by the model on load
        self._model: object | None = None
        self._load_lock = threading.Lock()
        self._closed = threading.Event()
        # single worker serialises synthesis calls on one model instance
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts"
        )

    @property
    def model_name(self) -> str:
        return f"phonon-31m-en/{self._voice}"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        """Load the model on the engine's executor thread. Blocking; call off the event loop."""
        self._executor.submit(self._load).result()

    def _load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            import ptts

            logger.info("Loading Phonon TTS model: %s", self._model_id)
            t0 = time.perf_counter()
            model = ptts.TTS(self._model_id, lang="en", quant=self._quant)
            self._sample_rate = model.sample_rate
            self._model = model
            logger.info(
                "Phonon TTS loaded in %.0fms (quant=%s)",
                (time.perf_counter() - t0) * 1000, getattr(model, "quant", "?"),
            )

    def _ensure_loaded(self):
        # Lazy path for direct callers; may run on the caller's thread.
        if self._closed.is_set():
            raise RuntimeError("TTS engine is closed")
        if self._model is None:
            self._load()
        return self._model

    def _fail(self, t0: float, error: str) -> TTSResult:
        return TTSResult(
            success=False, audio=None, sample_rate=self._sample_rate,
            channels=1, duration_seconds=0.0,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            model=self.model_name, error=error,
        )

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize complete audio from text."""
        t0 = time.perf_counter()
        try:
            audio = np.asarray(
                self._ensure_loaded().synth(text, voice=self._voice), dtype=np.float32
            )
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
            return self._fail(t0, str(exc))
        if audio.size == 0:
            return self._fail(t0, "no audio generated")
        return TTSResult(
            success=True,
            audio=audio,
            sample_rate=self._sample_rate,
            channels=1,
            duration_seconds=len(audio) / self._sample_rate,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            model=self.model_name,
        )

    def synthesize_stream(
        self, text: str, *, cancel: threading.Event | None = None
    ) -> Iterator[AudioChunk]:
        """Yield chunks as Phonon's decoder produces them, with no lookahead.

        Stops when `cancel` is set or the engine is closed. The underlying
        AudioStream is closed on every exit path. Raises TTSError on failure.
        """
        try:
            stream = self._ensure_loaded().stream(text, voice=self._voice)
        except Exception as exc:
            raise TTSError(f"TTS stream failed to start: {exc}") from exc
        try:
            for pcm in stream:
                if self._closed.is_set() or (cancel is not None and cancel.is_set()):
                    return
                yield AudioChunk(audio=np.asarray(pcm, dtype=np.float32), sample_rate=self._sample_rate)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"TTS streaming synthesis failed: {exc}") from exc
        finally:
            stream.close()

    async def stream_to(
        self,
        text: str,
        sink: Callable[[AudioChunk], bool],
        *,
        cancel: threading.Event | None = None,
    ) -> TTSStreamStats:
        """Synthesize on the engine thread, calling sink(chunk) there per chunk."""
        t0 = time.perf_counter()
        abort = threading.Event()  # set if the awaiting task is cancelled
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor, self._stream_worker, text, sink, cancel, abort, t0
            )
        except asyncio.CancelledError:
            abort.set()
            raise

    def _stream_worker(
        self,
        text: str,
        sink: Callable[[AudioChunk], bool],
        cancel: threading.Event | None,
        abort: threading.Event,
        t0: float,
    ) -> TTSStreamStats:
        first_ms: float | None = None
        chunks = 0
        seconds = 0.0
        stopped = abort.is_set()
        if not stopped:
            gen = self.synthesize_stream(text, cancel=cancel)
            try:
                for chunk in gen:
                    if abort.is_set():
                        stopped = True
                        break
                    if first_ms is None:
                        first_ms = (time.perf_counter() - t0) * 1000
                    chunks += 1
                    seconds += len(chunk.audio) / chunk.sample_rate
                    if not sink(chunk):
                        stopped = True
                        break
            finally:
                gen.close()
        synth_ms = (time.perf_counter() - t0) * 1000
        logger.info("first chunk %s ms, synth %.0f ms, %.1fs audio%s: %r",
                    f"{first_ms:.0f}" if first_ms is not None else "-", synth_ms, seconds,
                    " (stopped)" if stopped else "", text[:80])
        return TTSStreamStats(
            first_chunk_ms=first_ms,
            synth_ms=synth_ms,
            audio_seconds=seconds,
            chunks=chunks,
            cancelled=stopped or (cancel is not None and cancel.is_set()) or self._closed.is_set(),
        )

    async def synthesize_async(self, text: str) -> TTSResult:
        """Async wrapper that runs synthesis off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.synthesize, text)

    def close(self) -> None:
        """Stop any in-flight stream at its next chunk and release the model."""
        self._closed.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._model = None
