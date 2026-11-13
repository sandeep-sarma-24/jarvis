"""Voice Activity Detection — Silero VAD wrapper.

Responsibility: "Is this chunk speech?" Returns probability + latency.
Does NOT do segmentation — that belongs to segmentation.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    is_speech: bool
    confidence: float
    latency_ms: float


class VoiceActivityDetector:
    """Wraps Silero VAD (ONNX). process_chunk() is the only public entry point."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model: object | None = None
        self._torch: object | None = None
        self._sample_rate = 16000

    def load(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        logger.info("Loading Silero VAD (ONNX)…")
        t0 = time.perf_counter()
        self._model = load_silero_vad(onnx=True)
        self._torch = torch  # imported once here, not per chunk
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Silero VAD loaded in %.0f ms", elapsed)

    def process_chunk(self, chunk: np.ndarray) -> VADResult:
        """Process a single audio chunk (512 samples at 16 kHz).

        Returns VADResult with speech probability.
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded — call load() first")

        t0 = time.perf_counter()
        # Zero-copy for float32 chunks; .float() is a no-op then.
        tensor = self._torch.from_numpy(chunk).float()
        prob = self._model(tensor, self._sample_rate).item()
        latency = (time.perf_counter() - t0) * 1000

        return VADResult(
            is_speech=prob >= self._threshold,
            confidence=prob,
            latency_ms=latency,
        )

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()
