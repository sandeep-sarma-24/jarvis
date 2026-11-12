"""Speech recognition — IBM Granite Speech 5.0 TurboCTC via mlx-audio.

Responsibility: speech segment → transcript text.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass

import numpy as np

from app.config import ASRConfig

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    text: str
    language: str
    latency_ms: float


class SpeechRecognizer:
    """Wraps an mlx-audio STT model. Model loaded once at startup.

    MLX GPU streams are thread-local, so all inference runs on a single
    dedicated thread to avoid "no Stream in current thread" errors.
    """

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._session: object | None = None
        # ponytail: single-thread executor keeps MLX stream affinity
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="asr"
        )

    def load(self) -> None:
        from mlx_audio.stt.utils import load_model

        logger.info("Loading ASR model %s…", self._config.model)
        t0 = time.perf_counter()
        # Load on the dedicated thread so the MLX stream is created there
        self._session = self._executor.submit(
            load_model, self._config.model
        ).result()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("ASR model loaded in %.0f ms", elapsed)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """Transcribe a completed speech segment (numpy float32, 16 kHz mono).

        Must be called via transcribe_async() or on the dedicated executor thread.
        """
        if self._session is None:
            raise RuntimeError("ASR model not loaded — call load() first")

        t0 = time.perf_counter()
        if sample_rate != 16000:
            raise ValueError(f"ASR expects 16 kHz audio, got {sample_rate}")
        result = self._session.generate(audio)
        latency = (time.perf_counter() - t0) * 1000

        return ASRResult(
            text=result.text.strip(),
            language=getattr(result, "language", ""),
            latency_ms=latency,
        )

    async def transcribe_async(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> ASRResult:
        """Async wrapper that runs transcription on the dedicated MLX thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.transcribe, audio, sample_rate
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False)
