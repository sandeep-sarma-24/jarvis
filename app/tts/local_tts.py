"""Local TTS implementation using Kokoro-82M via mlx-audio.

Runs entirely on-device via Apple Silicon MLX. No network required
after initial model download (~327 MB from HuggingFace Hub).

Synchronous — designed to be called via asyncio.to_thread.
"""
from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np

from app.tts.model import AudioChunk, TTSResult

logger = logging.getLogger(__name__)


class KokoroTTS:
    """Kokoro-82M TTS via mlx-audio.

    ponytail: wraps mlx_audio Kokoro model. Upgrade path: swap to
    Qwen3-TTS/Holler for voice cloning if Kokoro voice quality
    is insufficient.
    """

    def __init__(
        self,
        *,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "bf_emma",
        speed: float = 1.0,
        sample_rate: int = 24000,
    ) -> None:
        self._model_id = model_id
        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return f"kokoro-82m/{self._voice}"

    def load(self) -> None:
        """Load the model. Downloads weights on first call."""
        from mlx_audio.tts.generate import load_model

        logger.info("Loading Kokoro TTS model: %s", self._model_id)
        t0 = time.perf_counter()
        self._model = load_model(self._model_id)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Kokoro TTS loaded in %.0fms", elapsed)

    def _ensure_loaded(self):
        if self._model is None:
            self.load()
        return self._model

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize complete audio from text."""
        model = self._ensure_loaded()

        t0 = time.perf_counter()
        try:
            # lang_code='a' uses American English G2P (works without espeak)
            # Voice style (bf_emma = British female) is in the model weights
            results = list(model.generate(  # type: ignore[union-attr]
                text,
                voice=self._voice,
                speed=self._speed,
                lang_code="a",
            ))
            latency = (time.perf_counter() - t0) * 1000

            if not results:
                return TTSResult(
                    success=False, audio=None, sample_rate=self._sample_rate,
                    channels=1, duration_seconds=0.0, latency_ms=round(latency, 1),
                    model=self.model_name, error="no audio generated",
                )

            # Concatenate all result chunks
            audio_arrays = [np.array(r.audio) for r in results if r.audio is not None]
            if not audio_arrays:
                return TTSResult(
                    success=False, audio=None, sample_rate=self._sample_rate,
                    channels=1, duration_seconds=0.0, latency_ms=round(latency, 1),
                    model=self.model_name, error="all chunks empty",
                )

            audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
            duration = len(audio) / self._sample_rate

            return TTSResult(
                success=True,
                audio=audio,
                sample_rate=self._sample_rate,
                channels=1,
                duration_seconds=duration,
                latency_ms=round(latency, 1),
                model=self.model_name,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("TTS synthesis failed: %s", exc)
            return TTSResult(
                success=False, audio=None, sample_rate=self._sample_rate,
                channels=1, duration_seconds=0.0, latency_ms=round(latency, 1),
                model=self.model_name, error=str(exc),
            )

    def synthesize_stream(self, text: str) -> Iterator[AudioChunk]:
        """Synthesize text and yield audio chunks per sentence.

        mlx-audio's Kokoro model.generate() yields results per text segment.
        """
        model = self._ensure_loaded()

        try:
            results = list(model.generate(  # type: ignore[union-attr]
                text,
                voice=self._voice,
                speed=self._speed,
                lang_code="a",
            ))

            for i, r in enumerate(results):
                if r.audio is not None:
                    audio_np = np.array(r.audio)
                    yield AudioChunk(
                        audio=audio_np,
                        sample_rate=self._sample_rate,
                        is_last=(i == len(results) - 1),
                    )
        except Exception as exc:
            logger.error("TTS streaming synthesis failed: %s", exc)

    def close(self) -> None:
        self._model = None
