"""Local TTS implementation using Gradium Phonon 31m (English) via ptts.

Runs entirely on-device (Rust runtime, no PyTorch). No network required
after initial model download (~65 MB q8 GGUF from HuggingFace Hub).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import Iterator

import numpy as np

from app.tts.model import AudioChunk, TTSResult

logger = logging.getLogger(__name__)


class PhononTTS:
    """Phonon 31m English TTS via the ptts package."""

    def __init__(
        self,
        *,
        model_id: str = "hf://gradium/phonon-31m-en",
        voice: str = "Easton",
        quant: str = "q8",
    ) -> None:
        self._model_id = model_id
        self._voice = voice
        self._quant = quant
        self._sample_rate = 24000  # overwritten by the model on load
        self._model: object | None = None
        # single worker serialises synthesis calls on one model instance
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts"
        )

    @property
    def model_name(self) -> str:
        return f"phonon-31m-en/{self._voice}"

    def load(self) -> None:
        """Load the model. Downloads weights on first call."""
        import ptts

        logger.info("Loading Phonon TTS model: %s", self._model_id)
        t0 = time.perf_counter()
        self._model = ptts.TTS(self._model_id, lang="en", quant=self._quant)
        self._sample_rate = self._model.sample_rate
        logger.info("Phonon TTS loaded in %.0fms", (time.perf_counter() - t0) * 1000)

    def _ensure_loaded(self):
        if self._model is None:
            self.load()
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

    def synthesize_stream(self, text: str) -> Iterator[AudioChunk]:
        """Yield audio chunks as Phonon's decoder produces them."""
        model = self._ensure_loaded()
        try:
            with model.stream(text, voice=self._voice) as chunks:
                prev = None
                for pcm in chunks:
                    if prev is not None:
                        yield AudioChunk(audio=prev, sample_rate=self._sample_rate)
                    prev = np.asarray(pcm, dtype=np.float32)
                if prev is not None:
                    yield AudioChunk(audio=prev, sample_rate=self._sample_rate, is_last=True)
        except Exception as exc:
            logger.error("TTS streaming synthesis failed: %s", exc)

    async def synthesize_async(self, text: str) -> TTSResult:
        """Async wrapper that runs synthesis off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.synthesize, text)

    def close(self) -> None:
        self._model = None
        self._executor.shutdown(wait=False)
