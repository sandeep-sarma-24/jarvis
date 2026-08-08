"""JARVIS — Microphone → VAD → ASR → Attention → Context → LLM → Response Decision.

Entry point and lifecycle orchestrator.
Phase 1 audio pipeline + Phase 2 attention + Phase 3 reasoning.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid

from app.config import JarvisConfig, load_config
from app.events.bus import EventBus
from app.events.types import Event, Modality, SpeechEvent
from app.audio.capture import MicrophoneCapture
from app.audio.vad import VoiceActivityDetector
from app.audio.segmentation import SpeechSegmenter
from app.audio.asr import SpeechRecognizer
from app.attention.engine import AttentionEngine
from app.memory.working import WorkingMemory

logger = logging.getLogger("jarvis")

# ── Metrics ──────────────────────────────────────────────────────────────

_metrics = {
    "audio_chunks_received": 0,
    "audio_chunks_dropped": 0,
    "speech_segments_detected": 0,
    "speech_segments_transcribed": 0,
    "speech_segments_failed": 0,
    "events_published": 0,
    "events_dropped": 0,
    "buffer_max_depth": 0,
    "reasoning_requests": 0,
    "reasoning_successes": 0,
    "reasoning_failures": 0,
    "reasoning_dropped": 0,
    "responses_generated": 0,
}


# ── Console subscriber ──────────────────────────────────────────────────

async def _console_handler(event: Event) -> None:
    if not isinstance(event, SpeechEvent):
        return
    ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
    ms = int((event.timestamp % 1) * 1000)
    md = event.metadata
    asr_ms = md.get("asr_latency_ms", "?")
    total_ms = md.get("total_speech_to_event_ms", "?")
    print(
        f"\n[{ts}.{ms:03d}] SPEECH\n"
        f"  event_id={event.id[:12]}…\n"
        f"  duration={event.duration_seconds:.2f}s\n"
        f"  confidence={event.confidence:.2f}\n"
        f"  asr_latency={asr_ms}ms\n"
        f"  speech_to_event={total_ms}ms\n"
        f'\n  "{event.transcript}"\n'
    )


# ── Pipeline ─────────────────────────────────────────────────────────────

async def _audio_pipeline(
    mic: MicrophoneCapture,
    vad: VoiceActivityDetector,
    segmenter: SpeechSegmenter,
    asr: SpeechRecognizer,
    bus: EventBus,
    session_id: str,
) -> None:
    """Async loop: read chunks → VAD → segment → ASR → publish SpeechEvent."""
    while True:
        chunk = await mic.buffer.get()
        _metrics["audio_chunks_received"] = mic.buffer.chunks_received
        _metrics["audio_chunks_dropped"] = mic.buffer.chunks_dropped
        _metrics["buffer_max_depth"] = mic.buffer.max_depth

        vad_result = vad.process_chunk(chunk)

        segment = segmenter.process(chunk, vad_result)
        if segment is None:
            continue

        _metrics["speech_segments_detected"] = segmenter.segments_detected

        depth_before = mic.buffer.depth
        drops_before = mic.buffer.chunks_dropped

        try:
            asr_result = await asyncio.to_thread(
                asr.transcribe, segment.audio, segment.sample_rate
            )
        except Exception:
            _metrics["speech_segments_failed"] += 1
            logger.exception("ASR failed on segment (%.2fs)", segment.duration_seconds)
            continue

        depth_after = mic.buffer.depth
        drops_after = mic.buffer.chunks_dropped

        if not asr_result.text:
            continue

        _metrics["speech_segments_transcribed"] += 1

        total_ms = (time.time() - segment.end_time) * 1000

        event = SpeechEvent(
            session_id=session_id,
            correlation_id=uuid.uuid4().hex,
            source=Modality.AUDIO,
            confidence=0.0,
            transcript=asr_result.text,
            duration_seconds=segment.duration_seconds,
            metadata={
                "asr_latency_ms": round(asr_result.latency_ms, 1),
                "total_speech_to_event_ms": round(total_ms, 1),
                "buffer_depth_before_asr": depth_before,
                "buffer_depth_after_asr": depth_after,
                "chunks_dropped_during_asr": drops_after - drops_before,
            },
        )

        await bus.publish(event)
        _metrics["events_published"] = bus.events_published
        _metrics["events_dropped"] = bus.events_dropped


# ── Reasoning pipeline ──────────────────────────────────────────────────

def _build_reasoning_handler(cfg: JarvisConfig, attention: AttentionEngine):
    """Build the attention→context→reasoning→response handler.

    Returns the async handler and the model instance (for cleanup).
    Uses a bounded semaphore to limit concurrent reasoning.
    """
    from app.context.builder import ContextBuilder
    from app.reasoning.local_model import OllamaModel
    from app.reasoning.model import ReasoningModel
    from app.response.decision import decide

    model: ReasoningModel = OllamaModel(
        model=cfg.reasoning.model,
        base_url=cfg.reasoning.ollama_url,
        timeout_seconds=cfg.reasoning.timeout_seconds,
    )

    context_builder = ContextBuilder(
        max_messages=cfg.reasoning.max_context_messages,
        max_characters=cfg.reasoning.max_context_characters,
    )

    reasoning_sem = asyncio.Semaphore(cfg.reasoning.max_concurrent_requests)
    reasoning_logger = logging.getLogger("jarvis.reasoning")

    async def _handler(event: Event) -> None:
        if not isinstance(event, SpeechEvent):
            return

        decision = attention.evaluate(event)

        # Console-style attention output
        if decision.should_reason:
            print(
                f"\n[ATTENTION]\n"
                f"  event_id={event.id[:12]}\n"
                f"  should_reason=true\n"
                f"  salience={decision.salience:.2f}\n"
            )
        else:
            print(
                f"\n[ATTENTION]\n"
                f"  event_id={event.id[:12]}\n"
                f"  should_reason=false\n"
                f"\n[NO RESPONSE]\n"
            )
            return

        if not cfg.reasoning.enabled or not cfg.response.enabled:
            return

        # Bounded concurrency: if semaphore is exhausted, drop the request
        if reasoning_sem.locked():
            _metrics["reasoning_dropped"] += 1
            reasoning_logger.warning(
                "Reasoning slot full, dropping event %s", event.id[:12]
            )
            return

        _metrics["reasoning_requests"] += 1

        async with reasoning_sem:
            ctx = context_builder.build(event, attention.working_memory)
            messages = context_builder.to_messages(ctx)

            reasoning_logger.debug(
                "Context: %d messages, %d chars",
                len(messages), ctx.total_characters,
            )

            result = await asyncio.to_thread(
                model.generate,
                messages,
                max_tokens=cfg.reasoning.max_output_tokens,
                temperature=cfg.reasoning.temperature,
            )

        if result.success:
            _metrics["reasoning_successes"] += 1
        else:
            _metrics["reasoning_failures"] += 1

        reasoning_logger.info(
            "REASONING event_id=%s model=%s latency_ms=%.0f success=%s "
            "context_messages=%d context_chars=%d",
            event.id[:12], result.model, result.latency_ms, result.success,
            len(messages), ctx.total_characters,
        )

        resp = decide(event.id, result)

        if resp.respond and resp.text:
            _metrics["responses_generated"] += 1
            print(
                f"\n[REASONING]\n"
                f"  model={result.model}\n"
                f"  latency={result.latency_ms:.0f}ms\n"
                f"\n[RESPONSE]\n"
                f"  {resp.text}\n"
            )
        else:
            print(
                f"\n[RESPONSE_DECISION]\n"
                f"  event_id={event.id[:12]}\n"
                f"  respond=false\n"
                f"  reason=\"{resp.reason}\"\n"
            )

        reasoning_logger.info(
            "RESPONSE_DECISION event_id=%s respond=%s reason=%r",
            event.id[:12], resp.respond, resp.reason,
        )

    return _handler, model


# ── Lifecycle ────────────────────────────────────────────────────────────

async def run(cfg: JarvisConfig | None = None) -> None:
    cfg = cfg or load_config()

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    session_id = uuid.uuid4().hex
    logger.info("JARVIS session %s starting", session_id[:12])
    logger.info("Microphone capture will be active — audio is processed locally, never stored.")

    # 1. EventBus
    bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
    bus.subscribe(SpeechEvent, _console_handler)

    # 2. Attention + Reasoning pipeline (Phase 2 + 3)
    attention: AttentionEngine | None = None
    model_cleanup = None

    if cfg.attention.enabled:
        working_mem = WorkingMemory(max_size=cfg.attention.working_memory_size)
        attention = AttentionEngine(
            assistant_names=cfg.attention.assistant_names,
            reason_threshold=cfg.attention.reason_threshold,
            working_memory=working_mem,
        )

        if cfg.reasoning.enabled:
            handler, model_cleanup = _build_reasoning_handler(cfg, attention)
            bus.subscribe(SpeechEvent, handler)
            logger.info(
                "Reasoning pipeline active — model=%s max_concurrent=%d",
                cfg.reasoning.model, cfg.reasoning.max_concurrent_requests,
            )
        else:
            async def _attention_only_handler(event: Event) -> None:
                if isinstance(event, SpeechEvent) and attention is not None:
                    attention.evaluate(event)
            bus.subscribe(SpeechEvent, _attention_only_handler)

        logger.info("Attention engine active — names=%s threshold=%.2f",
                     cfg.attention.assistant_names, cfg.attention.reason_threshold)

    await bus.start()

    # 3. VAD
    vad = VoiceActivityDetector(threshold=cfg.vad.threshold)
    vad.load()

    # 4. ASR
    asr = SpeechRecognizer(cfg.asr)
    asr.load()

    # 5. Segmenter
    segmenter = SpeechSegmenter(
        config=cfg.vad,
        sample_rate=cfg.audio.sample_rate,
        chunk_samples=cfg.audio.chunk_samples,
    )

    # 6. Microphone
    mic = MicrophoneCapture(cfg.audio)
    loop = asyncio.get_running_loop()

    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    mic.start(loop)

    # 7. Run pipeline
    pipeline_task = asyncio.create_task(
        _audio_pipeline(mic, vad, segmenter, asr, bus, session_id)
    )

    logger.info("Pipeline running — speak into the microphone. Ctrl+C to stop.")

    await shutdown.wait()

    # 8. Graceful shutdown
    logger.info("Shutting down…")
    mic.stop()
    pipeline_task.cancel()
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass
    await bus.shutdown()

    if model_cleanup is not None:
        model_cleanup.close()

    logger.info(
        "Session %s ended — chunks=%d dropped=%d segments=%d transcribed=%d failed=%d max_depth=%d",
        session_id[:12],
        _metrics["audio_chunks_received"],
        _metrics["audio_chunks_dropped"],
        _metrics["speech_segments_detected"],
        _metrics["speech_segments_transcribed"],
        _metrics["speech_segments_failed"],
        _metrics["buffer_max_depth"],
    )
    if attention is not None:
        logger.info("Attention counters: %s", attention.counters)
    logger.info(
        "Reasoning: requests=%d success=%d fail=%d dropped=%d responses=%d",
        _metrics["reasoning_requests"],
        _metrics["reasoning_successes"],
        _metrics["reasoning_failures"],
        _metrics["reasoning_dropped"],
        _metrics["responses_generated"],
    )


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
