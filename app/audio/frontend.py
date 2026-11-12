"""Audio frontend: mic buffer → VAD → segmentation → ASR → Transcript.

See docs/architecture/06-engineering-contracts.md section 2.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from collections import deque
from typing import Callable

from app.audio.segmentation import SpeechSegment
from app.config import BargeInConfig
from app.contracts import Transcript

logger = logging.getLogger(__name__)

_SEGMENT_QUEUE_MAX = 2


class AudioFrontend:
    """Consumes the mic continuously; ASR runs concurrently on finished segments."""

    def __init__(self, mic, vad, segmenter, asr, *,
                 on_transcript: Callable[[Transcript], None],
                 on_barge_in: Callable[[float], None] | None = None,
                 barge_in: BargeInConfig, playback_tail_ms: int = 200) -> None:
        self._mic = mic
        self._vad = vad
        self._segmenter = segmenter
        self._asr = asr
        self._on_transcript = on_transcript
        self._on_barge_in = on_barge_in
        self._barge_in = barge_in
        self._tail_s = playback_tail_ms / 1000

        # Silero can exceed ~2 ms per chunk under CPU load, so VAD runs off the
        # event loop. One dedicated thread keeps calls serial and ordered and
        # never queues behind other asyncio.to_thread work.
        self._vad_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vad"
        )
        # (segment, emitted wall time); deque(maxlen) drops the oldest.
        self._segments: deque[tuple[SpeechSegment, float]] = deque(maxlen=_SEGMENT_QUEUE_MAX)
        self._segment_ready = asyncio.Event()

        self._playback_active = False
        self._resume_at = 0.0  # capture time before which audio is discarded
        self._gated = False
        self._barge_samples = 0
        self._barge_fired = False

        self.metrics: dict[str, float] = {
            "audio_chunks_received": 0,
            "audio_chunks_dropped": 0,
            "buffer_max_depth": 0,
            "chunks_discarded_playback": 0,
            "speech_segments_detected": 0,
            "speech_segments_dropped": 0,
            "speech_segments_transcribed": 0,
            "speech_segments_failed": 0,
            "barge_ins": 0,
        }

    def set_playback_active(self, active: bool) -> None:
        """Call from the loop thread when playback starts (True) or ends (False)."""
        if active and not self._playback_active:
            self._barge_fired = False
            self._barge_samples = 0
        if not active and self._playback_active:
            self._resume_at = time.time() + self._tail_s
            while self._mic.buffer.get_nowait() is not None:
                self.metrics["chunks_discarded_playback"] += 1
        self._playback_active = active

    async def run(self) -> None:
        asr_task = asyncio.create_task(self._asr_loop())
        try:
            await self._consume()
        finally:
            asr_task.cancel()
            self._vad_executor.shutdown(wait=False, cancel_futures=True)

    async def _run_vad(self, chunk):
        return await asyncio.get_running_loop().run_in_executor(
            self._vad_executor, self._vad.process_chunk, chunk
        )

    async def _consume(self) -> None:
        buf = self._mic.buffer
        while True:
            chunk, t = await buf.get()
            self.metrics["audio_chunks_received"] = buf.chunks_received
            self.metrics["audio_chunks_dropped"] = buf.chunks_dropped
            self.metrics["buffer_max_depth"] = buf.max_depth

            gated = self._playback_active or t < self._resume_at
            if gated != self._gated:
                # Entering or leaving playback: never join audio across it.
                self._vad.reset()
                self._segmenter.reset()
                self._gated = gated
            if gated:
                if self._playback_active and self._barge_in.enabled:
                    await self._check_barge_in(chunk, t)
                else:
                    self.metrics["chunks_discarded_playback"] += 1
                continue

            segment = self._segmenter.process(chunk, await self._run_vad(chunk), t)
            if segment is None:
                continue
            self.metrics["speech_segments_detected"] += 1
            if len(self._segments) == _SEGMENT_QUEUE_MAX:
                self.metrics["speech_segments_dropped"] += 1
                logger.warning("ASR behind: dropping oldest queued segment")
            self._segments.append((segment, time.time()))
            self._segment_ready.set()

    async def _check_barge_in(self, chunk, t: float) -> None:
        result = await self._run_vad(chunk)
        if result.confidence < self._barge_in.vad_threshold:
            self._barge_samples = 0
            return
        self._barge_samples += len(chunk)
        needed = self._barge_in.min_speech_ms * self._segmenter.sample_rate / 1000
        if not self._barge_fired and self._barge_samples >= needed:
            self._barge_fired = True
            self.metrics["barge_ins"] += 1
            if self._on_barge_in is not None:
                self._on_barge_in(t)

    async def _asr_loop(self) -> None:
        buf = self._mic.buffer
        while True:
            if not self._segments:
                self._segment_ready.clear()
                await self._segment_ready.wait()
                continue
            segment, emitted = self._segments.popleft()
            depth_before, drops_before = buf.depth, buf.chunks_dropped
            try:
                # transcribe_async runs on the ASR's own single thread (MLX affinity).
                result = await self._asr.transcribe_async(segment.audio, segment.sample_rate)
            except Exception:
                self.metrics["speech_segments_failed"] += 1
                logger.exception("ASR failed on segment (%.2fs)", segment.duration_seconds)
                continue
            if not result.text:
                continue
            self.metrics["speech_segments_transcribed"] += 1
            transcript = Transcript(
                text=result.text,
                speech_start_time=segment.start_time,
                speech_end_time=segment.speech_end_time,
                segment_emitted_time=emitted,
                asr_latency_ms=result.latency_ms,
                audio_seconds=segment.duration_seconds,
                metadata={
                    "buffer_depth_before_asr": depth_before,
                    "buffer_depth_after_asr": buf.depth,
                    "chunks_dropped_during_asr": buf.chunks_dropped - drops_before,
                    "segments_dropped": self.metrics["speech_segments_dropped"],
                },
            )
            try:
                self._on_transcript(transcript)
            except Exception:
                logger.exception("on_transcript callback failed")
