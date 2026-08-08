"""Speech recognition — Qwen3-ASR wrapper via mlx-qwen3-asr.

Responsibility: speech segment → transcript text.
"""
from __future__ import annotations

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
    """Wraps mlx-qwen3-asr. Model loaded once at startup."""

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._session: object | None = None

    def load(self) -> None:
        from mlx_qwen3_asr import Session

        logger.info("Loading ASR model %s…", self._config.model)
        t0 = time.perf_counter()
        self._session = Session(model=self._config.model)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("ASR model loaded in %.0f ms", elapsed)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """Transcribe a completed speech segment (numpy float32, 16 kHz mono)."""
        if self._session is None:
            raise RuntimeError("ASR model not loaded — call load() first")

        t0 = time.perf_counter()
        result = self._session.transcribe((audio, sample_rate))
        latency = (time.perf_counter() - t0) * 1000

        return ASRResult(
            text=result.text.strip(),
            language=getattr(result, "language", ""),
            latency_ms=latency,
        )
